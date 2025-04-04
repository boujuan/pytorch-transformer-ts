# %%
# %matplotlib inline

import multiprocessing
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from itertools import islice


# %%
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
# from gluonts.dataset.repository.datasets import get_dataset
from gluonts.multivariate.datasets.dataset import datasets 
from estimator import InformerEstimator
from lightning_module import InformerLightningModule
from lightning.pytorch import seed_everything

from gluonts.model.forecast_generator import SampleForecastGenerator, DistributionForecastGenerator
from gluonts.torch.distributions import LowRankMultivariateNormalOutput

# checkpoint = '/Users/ahenry/Documents/toolboxes/pytorch-transformer-ts/lightning_logs/version_88/checkpoints/epoch=0-step=10.ckpt'
# checkpoint = '/Users/ahenry/Documents/toolboxes/pytorch-transformer-ts/lightning_logs/version_18/checkpoints/epoch=42-step=4300.ckpt' # distributionforecast checkpoint
checkpoint = '/Users/ahenry/Documents/toolboxes/pytorch-transformer-ts/lightning_logs/version_101/checkpoints/epoch=0-step=100.ckpt' # sampleforecats checkpoint
# %%
# torch.manual_seed(0)
# np.random.seed(0)
seed_everything(0, workers=True)
# %%

if False:
    # dataset = get_dataset("electricity")
    dataset = datasets["solar"]()
    training_data = dataset.train_ds
    test_data = dataset.test_ds
    freq = dataset.freq
    prediction_length = dataset.prediction_length
    context_length = dataset.prediction_length*7
    cardinality = None
    embedding_dimension = None
    num_feat_static_cat = 1
    input_size = dataset.target_dim
    num_feat_dynamic_real = 0
    num_feat_static_real = 0
    
    
    # d = 0
    # ds = train_dataset[d] 
    # training_data = [{
    #         "target": IterableLazyFrame(data=pd.DataFrame(ds["target"].T, columns=np.arange(ds["target"].shape[0])), dtype=pl.Float32, load=True),
    #         **{k: ds[k] for k in ds if k != "target"}
    #         } for d, ds in enumerate(train_dataset)]

    # test_data = [{
    #         "target": IterableLazyFrame(data=pd.DataFrame(ds["target"].T, columns=np.arange(ds["target"].shape[0])), dtype=pl.Float32, load=True),
    #         **{k: ds[k] for k in ds if k != "target"}
    #         } for d, ds in enumerate(test_dataset)]
# else:
#     with open("/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml", 'r') as file:
#         config  = yaml.safe_load(file)
        
#     data_module = DataModule(data_path=config["dataset"]["data_path"], n_splits=config["dataset"]["n_splits"],
#                             continuity_groups=None, train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
#                                 val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
#                                 prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
#                                 target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
#                                 freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["target_turbine_ids"],
#                                     per_turbine_target=config["dataset"]["per_turbine_target"], dtype=pl.Float32)
#     # if RUN_ONCE:
#     data_module.generate_splits()
#     train_dataset = data_module.train_dataset
#     test_dataset = data_module.test_dataset
#     freq = data_module.freq
#     prediction_length = data_module.prediction_length
#     context_length = data_module.context_length
#     cardinality = data_module.cardinality
#     embedding_dimension = None
#     num_feat_dynamic_real = data_module.num_feat_dynamic_real
#     num_feat_static_cat = data_module.num_feat_static_cat
#     input_size = data_module.num_target_vars
#     num_feat_static_real = data_module.num_feat_static_real
else:
    import pandas as pd
    data_path = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/preprocessed_flasc_data/sample.csv"
    wf_dataset = pd.read_csv(data_path)
    train_split = 4000
    freq = "60s"
    prediction_length = 10
    context_length = 20
    wf_training_data = [{"item_id": "all_items",
                        "feat_static_cat": [0],
                        "target": wf_dataset.iloc[:train_split][[col for col in wf_dataset.columns if col.startswith("ws_")]].values.astype("float32").T,
                        "start": pd.Period(pd.to_datetime(wf_dataset.iloc[:train_split]["time"].iloc[0]), freq=freq),
                        "feat_dynamic_real": wf_dataset.iloc[:train_split][[col for col in wf_dataset.columns if col.startswith("nd_")]].values.astype("float32").T
                        }]
    wf_test_data = [{"item_id": "all_items",
                    "feat_static_cat": [0],
                        "target": wf_dataset.iloc[train_split:][[col for col in wf_dataset.columns if col.startswith("ws_")]].values.astype("float32").T,
                        "start": pd.Period(pd.to_datetime(wf_dataset.iloc[train_split:]["time"].iloc[0]), freq=freq),
                        "feat_dynamic_real": wf_dataset.iloc[train_split:][[col for col in wf_dataset.columns if col.startswith("nd_")]].values.astype("float32").T
                        }]
    input_size = len([col for col in wf_dataset.columns if col.startswith("ws_")])
    num_feat_dynamic_real = len([col for col in wf_dataset.columns if col.startswith("nd_")])
    cardinality = embedding_dimension = None
    num_feat_static_real = 0
# %%


# %%
# training_data

# %%
estimator = InformerEstimator(
    freq=freq,
    prediction_length=prediction_length,
    context_length=context_length,
    
    # 
    num_feat_static_cat=1,
    input_size=input_size,
    num_feat_static_real=num_feat_static_real,
    num_feat_dynamic_real=num_feat_dynamic_real,
    cardinality=cardinality,
    embedding_dimension=embedding_dimension,
    
    # attention hyper-params
    lags_seq=[0],
    dim_feedforward=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_model=64,
    n_heads=2,
    activation="relu",
    use_lazyframe=False,
    # training params
    batch_size=128,
    num_batches_per_epoch=100,
    trainer_kwargs=dict(max_epochs=1, accelerator='cpu', devices=1),
    distr_output=LowRankMultivariateNormalOutput(dim=input_size, rank=8)
)

# %%
TEST = True 

predictor = estimator.train(
    training_data=wf_training_data,
    # shuffle_buffer_length=1024,
    forecast_generator=DistributionForecastGenerator(estimator.distr_output),
    # forecast_generator=SampleForecastGenerator(estimator.distr_output),
    ckpt_path=checkpoint
)
if TEST:
    # %%
    model = InformerLightningModule.load_from_checkpoint(checkpoint)
    transformation = estimator.create_transformation(use_lazyframe=False)
    predictor = estimator.create_predictor(transformation, model, 
                                            forecast_generator=DistributionForecastGenerator(estimator.distr_output)
                                            # forecast_generator=SampleForecastGenerator(estimator.distr_output)
                                            )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=wf_test_data,
        predictor=predictor,
        output_distr_params={"loc": "mean", "cov_factor": "cov_factor", "cov_diag": "cov_diag"} # mapping from distr parameters (after transformation) to LowRankMultivariateNormalOutput (with rank=0) parameters 
        # output_distr_params=False
    )

    # %%
    forecasts = list(islice(forecast_it, 1))

    # %%
    tss = list(islice(ts_it, 1))

    # %%
    # num_workers is limited to 10 if cpu has more cores
    num_workers = min(multiprocessing.cpu_count(), 1)

    evaluator = MultivariateEvaluator(num_workers=None)

    # %%
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))

    # %%
    agg_metrics

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    date_formater = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})

    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        # ax = plt.subplot(3, 3, idx+1)
        # ax = plt.subplots(1, 1)

        # Convert index for plot
        # ts = ts[-4 * dataset.prediction_length:].to_timestamp()
        ts = ts.loc[ts.index.isin(forecast.index), :] 
        ax.plot(ts.index.to_timestamp(), ts.values, linestyle='--')
        ax.plot(forecast.index.to_timestamp(), forecast.distribution.loc)
        # ax.set_xticks(ts.index, rotation=60)
        ax.set_title(forecast.item_id)
        ax.xaxis.set_major_formatter(date_formater)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()


