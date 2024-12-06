# %%
import os
import numpy as np
import multiprocessing as mp
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from itertools import islice
from gluonts.evaluation import make_evaluation_predictions, MultivariateEvaluator 
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.transform.sampler import InstanceSampler
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.dataset.util import to_pandas
from gluonts.model.forecast_generator import DistributionForecastGenerator
from estimator import InformerEstimator
from lightning.pytorch.loggers import WandbLogger
import pandas as pd
from wind_forecasting.datasets.data_module import DataModule

# %%
if __name__ == "__main__":
    from wind_forecasting.datasets.wind_farm import KPWindFarm
    # from wind_forecasting.models.spacetimeformer.spacetimeformer.spacetimeformer_model import Spacetimeformer_Forecaster
    from sys import platform

    if platform == "darwin":
        LOG_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/"
        DATA_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/short_loaded_data_calibrated_filtered_split_imputed_normalized.parquet"
        NORM_CONSTS = pd.read_csv("/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/normalization_consts.csv", index_col=None)
        n_workers = mp.cpu_count()
        accelerator = "auto"
        devices = "auto"
        num_nodes = 1
        strategy = "auto"
        dataset_class = "KPWindFarm"
        model_class = "Spacetimeformer_Forecaster"
    elif platform == "linux":
        LOG_DIR = "/projects/ssc/ahenry/wind_forecasting/logging/"
        DATA_PATH = "/projects/ssc/ahenry/wind_forecasting/awaken_data/filled_data_calibrated_filtered_split_imputed_normalized.parquet"
        NORM_CONSTS = pd.read_csv("/projects/ssc/ahenry/wind_forecasting/awaken_data/normalization_consts.csv", index_col=None)
        n_workers = int(os.environ["SLURM_GPUS_ON_NODE"])
        accelerator = "auto"
        devices = 2
        num_nodes = 1
        strategy = "ddp_find_unused_parameters_true"
        dataset_class = "KPWindFarm"
        model_class = "Spacetimeformer_Forecaster"

    ## DEFINE CONFIGURATION
    config = {
        "experiment": {
            "run_name": "windfarm_debug",
            "log_dir": LOG_DIR
        },
        "dataset": {
            "dataset_class": dataset_class,
            "data_path": DATA_PATH,
            "normalization_consts": NORM_CONSTS,
            "context_len": 4, # 120=10 minutes for 5 sec sample size,
            "target_len":  3, # 120=10 minutes for 5 sec sample size,
            # "target_turbine_ids": ["wt029", "wt034", "wt074"],
            "normalize": False, 
            "batch_size": 128,
            "workers": n_workers,
            "overfit": False,
            "test_split": 0.15,
            "val_split": 0.15,
            "collate_fn": None,
            "dataset_kwargs": { # specific to class KPWindFarm or similar 
                "target_turbine_ids": ["wt029"] #, "wt034", "wt074"]
            }
        },
        "model": {
            "model_class": model_class,
            'embed_size': 32, # Determines dimension of the embedding space
            'num_layers': 3, # Number of transformer blocks stacked
            'heads': 4, # Number of heads for spatio-temporal attention
            'forward_expansion': 4, # Multiplier for feedforward network size
            'output_size': 1, # Number of output variables,
            "d_model": 5,
            "d_queries_keys": 5, 
            "d_values": 5, 
            "d_ff": 5
        },
        "callbacks": {
            "progress_bar": {}, 
            "early_stopping": {}, 
            "model_checkpoint": {}, 
            "lr_monitor": {True}
        },
        "trainer": {
            "grad_clip_norm": 0.0, # Prevents gradient explosion if > 0 
            "limit_val_batches": 1.0, 
            "val_check_interval": 1,
            "debug": False, 
            "accumulate": 1.0,
            "max_epochs": 100, # Maximum number of epochs to train
            # "precision": '32-true', # 16-mixed enables mixed precision training, 32-true is full precision
            # 'batch_size': 32, # larger = more stable gradients
            # 'lr': 0.0001, # Step size
            # 'dropout': 0.1, # Regularization parameter (prevents overfitting)
            # 'patience': 50, # Number of epochs to wait before early stopping
            # 'accumulate_grad_batches': 2, # Simulates a larger batch size
        }
    }

    ## SETUP LOGGING=
    # Initialize wandb only on rank 0
    # os.environ["WANDB_INIT_TIMEOUT"] = "600"
    # os.environ["WANDB_INIT_TIMEOUT"] = "300"
    # os.environ["WANDB_DEBUG"] = "true"

    wandb_logger = None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        if not os.path.exists(config["experiment"]["log_dir"]):
            os.makedirs(config["experiment"]["log_dir"])
        # wandb.login() # Login to wandb website
        # entity=aoife-henry-university-of-colorado-boulder
        wandb_logger = WandbLogger(
            project="wf_forecasting",
            name=config["experiment"]["run_name"],
            log_model=True,
            save_dir=config["experiment"]["log_dir"],
            config=config
        )

    ## CREATE DATASET
    data_module = DataModule(
            dataset_class=globals()[config["dataset"]["dataset_class"]],
            config=config
    )

    # %%
    df = pd.read_parquet(DATA_PATH).resample('30s', on="time").mean()
    df.index.rename("timestamp")
    # %%

    sub_df = df.loc[df["continuity_group"] == df["continuity_group"].value_counts().index[1]]\
            .drop(columns="continuity_group")
    sub_df = sub_df[[col for col in sub_df.columns if any(col.__contains__(tid) for tid in ["wt001", "wt002", "wt003"])]]
    target_cols = [col for col in sub_df.columns if any(prefix in col for prefix in ["ws_horz", "ws_vert"])]
    past_feat_dynamic_real = [col for col in sub_df.columns if any(prefix in col for prefix in ["nd_cos", "nd_sin"])]
    float64_cols = list(sub_df.select_dtypes(include="float64"))
    sub_df[float64_cols] = sub_df[float64_cols].astype("float32")
    sub_df.head(10)

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(sub_df[[col for col in sub_df.columns if "nd_cos" in col]])
    ax[0].set_title("nd_cos")
    ax[1].plot(sub_df[[col for col in sub_df.columns if "nd_sin" in col]])
    ax[1].set_title("nd_sin")
    ax[2].plot(sub_df[[col for col in sub_df.columns if "ws_horz" in col]])
    ax[2].set_title("ws_horz")
    ax[3].plot(sub_df[[col for col in sub_df.columns if "ws_vert" in col]])
    ax[3].set_title("ws_vert")

    # %%
    from gluonts.dataset.common import TrainDatasets, MetaData, BasicFeatureInfo, CategoricalFeatureInfo
    # dataset = get_dataset("electricity_nips", regenerate=False)
    # alternatively import as single pandas dataset
    # dataset = PandasDataset(
        # tgt_col: sub_df[feat_dynamic_real + target_cols].rename(columns={tgt_col: "target"}) 
        # sub_df[past_feat_dynamic_real + target_cols], 
        # feat_dynamic_real=past_feat_dynamic_real, target=target_cols, assume_sorted=True)

    static_features = pd.DataFrame(
        {
            "turbine_id": pd.Categorical(col.split("_")[-1] for col in target_cols),
            "output_category": pd.Categorical("_".join(col.split("_")[:-1]) for col in target_cols)
        },
        index=target_cols
    )
    # TODO should I add these categorical variable..., multivariategrouper removes the feat_static_features...
    n_turbines = static_features["turbine_id"].dtype.categories.shape[0]
    n_output_categories = static_features["output_category"].dtype.categories.shape[0]
    dataset = TrainDatasets(
        metadata=MetaData(freq=sub_df.index.freq.freqstr, feat_dynamic_real=[BasicFeatureInfo(name=feat) for feat in past_feat_dynamic_real],
                            feat_static_cat=[CategoricalFeatureInfo(name="turbine_id", cardinality=n_turbines),
                                            CategoricalFeatureInfo(name="output_category", cardinality=n_output_categories)], 
                            prediction_length=config["dataset"]["target_len"]), 
        train=PandasDataset(
            {tgt_col: sub_df.iloc[:-config["dataset"]["target_len"]][past_feat_dynamic_real + [tgt_col]]\
                            .rename(columns={tgt_col: "target"})
            for tgt_col in target_cols}, feat_dynamic_real=past_feat_dynamic_real, static_features=static_features, assume_sorted=True),
        test=PandasDataset(
            {tgt_col: sub_df[past_feat_dynamic_real + [tgt_col]]\
                            .rename(columns={tgt_col: "target"}) for tgt_col in target_cols}, 
            feat_dynamic_real=past_feat_dynamic_real, static_features=static_features, assume_sorted=True))

    n_output_vars = len(list(iter(dataset.train)))
    train_grouper = MultivariateGrouper(
        max_target_dim=int(n_output_vars)
    )

    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=int(n_output_vars),
    )
    train_dataset = train_grouper(dataset.train)
    test_dataset = test_grouper(dataset.test)

    # %% 
    
    # TODO add train_sampler and validation_sampler to change training_data_loader and validation_data_loader to split data based on continuity_group in PytorchEstimator
    
    from gluonts.time_feature._base import minute_of_hour
    estimator = InformerEstimator(
        freq=dataset.metadata.freq, 
        prediction_length=dataset.metadata.prediction_length,
        # context_length=config["dataset"]["context_len"], # CHANGE
        context_length=dataset.metadata.prediction_length * 2,
        num_feat_dynamic_real=len(dataset.metadata.feat_dynamic_real), #len(past_feat_dynamic_real),
        num_feat_static_cat=len(dataset.metadata.feat_static_cat),
        num_feat_static_real=len(dataset.metadata.feat_static_real),
        input_size=n_output_vars,
        scaling=False,
        dim_feedforward=32,
        d_model=64, # TODO QUESTION should be more than input size ?
        num_encoder_layers=2,
        num_decoder_layers=2,
        nhead=2,
        activation="relu",
        time_features=[minute_of_hour],
        distr_output=LowRankMultivariateNormalOutput(dim=n_output_vars, rank=2),
        trainer_kwargs=dict(max_epochs=1, limit_train_batches=100, accelerator='cpu', devices=1, logger=wandb_logger),
    )

    # %%
    # set forecast_generator to DistributionForecastGenerator to access mean and variance in InformerEstimator.create_predictor
    predictor = estimator.train(
        # training_data=dataset.train,
        training_data=train_dataset,
        forecast_generator=DistributionForecastGenerator(estimator.distr_output)
        # shuffle_buffer_length=1024
    )

    # %%
    forecast_it, ts_it = make_evaluation_predictions(
        # dataset=dataset.test, 
        dataset=test_dataset,
        predictor=predictor,
        output_distr_params=True
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # %%
    # num_workers is limited to 10 if cpu has more cores
    num_workers = min(mp.cpu_count(), 10)
    # TODO add custom evaluation functions
    # TODO ensure that pandas dataframe contains item_id 
    evaluator = MultivariateEvaluator(num_workers=num_workers)

    # %%
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts), num_series=n_output_vars)

    # %%
    agg_metrics

    # %%
    rows = 3
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(6, 12))
    axx = axs.ravel()
    seq_len, target_dim = tss[0].shape
    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        tss[0][-3 * dataset.metadata.prediction_length :][dim].plot(ax=ax)

        # (quantile, target_dim, seq_len)
        pred_df = pd.DataFrame(
            {q: forecasts[0].quantile(q)[dim] for q in [0.1, 0.5, 0.9]},
            index=forecasts[0].index,
        )

        ax.fill_between(
            forecasts[0].index, pred_df[0.1], pred_df[0.9], alpha=0.2, color='g'
        )
        pred_df[0.5].plot(ax=ax, color='g')
    plt.show()

    # plt.figure(figsize=(20, 15))
    # date_formater = mdates.DateFormatter('%b, %d')
    # plt.rcParams.update({'font.size': 15})

    # for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    #     ax = plt.subplot(3, 3, idx+1)

    #     # Convert index for plot
    #     ts = ts[-4 * dataset.metadata.prediction_length:].to_timestamp()
        
    #     plt.plot(ts, label="target")
    #     forecast.plot( color='g')
    #     plt.xticks(rotation=60)
    #     plt.title(forecast.item_id)
    #     ax.xaxis.set_major_formatter(date_formater)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()
