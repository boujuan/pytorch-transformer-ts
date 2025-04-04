# %%
# %matplotlib inline

import multiprocessing
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from itertools import islice


# %%
from gluonts.evaluation import make_evaluation_predictions, MultivariateEvaluator
# from gluonts.dataset.repository.datasets import get_dataset
from gluonts.multivariate.datasets.dataset import datasets 
from estimator import AutoformerEstimator
from lightning_module import AutoformerLightningModule
from lightning.pytorch import seed_everything

from gluonts.model.forecast_generator import SampleForecastGenerator, DistributionForecastGenerator
from gluonts.torch.distributions import LowRankMultivariateNormalOutput

# checkpoint = '/Users/ahenry/Documents/toolboxes/pytorch-transformer-ts/lightning_logs/version_65/checkpoints/epoch=0-step=10.ckpt'
checkpoint = "/Users/ahenry/Documents/toolboxes/pytorch-transformer-ts-og/lightning_logs/version_21/checkpoints/epoch=0-step=10.ckpt"
# %%
# torch.manual_seed(0)
# np.random.seed(0)
seed_everything(0, workers=True)
# %%

if True:
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

# %%


# %%
# training_data

# %%
estimator = AutoformerEstimator(
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
    dim_feedforward=8,
    num_encoder_layers=2,
    num_decoder_layers=1,
    n_heads=2,
    activation="relu",
    lags_seq=[1, 2], 
    use_lazyframe=False,
    # training params
    batch_size=128,
    num_batches_per_epoch=10,
    trainer_kwargs=dict(max_epochs=1, accelerator='cpu', devices=1),
    distr_output=LowRankMultivariateNormalOutput(dim=input_size, rank=8)
)

# %%
TEST = True 

predictor = estimator.train(
    training_data=training_data,
    # shuffle_buffer_length=1024,
    forecast_generator=DistributionForecastGenerator(estimator.distr_output),
    ckpt_path=checkpoint
)
if TEST:
    # %%
    model = AutoformerLightningModule.load_from_checkpoint(checkpoint)
    transformation = estimator.create_transformation(use_lazyframe=False)
    predictor = estimator.create_predictor(transformation, model, 
                                            forecast_generator=DistributionForecastGenerator(estimator.distr_output))
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,
        predictor=predictor,
        output_distr_params={"loc": "mean", "cov_factor": "cov_factor", "cov_diag": "cov_diag"}
    )

    # %%
    forecasts = list(islice(forecast_it, 1))

    # %%
    tss = list(islice(ts_it, 1))

    # %%
    # num_workers is limited to 10 if cpu has more cores
    num_workers = min(multiprocessing.cpu_count(), 10)

    evaluator = MultivariateEvaluator(num_workers=None)

    # %%
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))

    # %%
    agg_metrics

    # %%
    fig = plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})

    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx+1)

        # Convert index for plot
        # ts = ts[-4 * dataset.prediction_length:].to_timestamp()
        
        # ax.plot(ts.index.to_timestamp(), ts.values, linestyle='--')
        ax.plot(forecast.index.to_timestamp(), forecast.distribution.loc)
        # ax.set_xticks(ts.index, rotation=60)
        ax.set_title(forecast.item_id)
        ax.xaxis.set_major_formatter(date_formater)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()



