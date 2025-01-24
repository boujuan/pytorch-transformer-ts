# %%
import os
from collections import defaultdict
from itertools import islice, cycle, chain
import multiprocessing as mp
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from matplotlib import colormaps, dates as mdates
import seaborn as sns

from gluonts.dataset.repository.datasets import get_dataset

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions

from lightning.pytorch.loggers import WandbLogger
import wandb

# from wind_forecasting.datasets.data_module import DataModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from wind_forecasting.preprocessing.data_module import DataModule

# %%
if __name__ == "__main__":
    TEST = False
    TRAIN = True

    # from wind_forecasting.models.spacetimeformer.spacetimeformer.spacetimeformer_model import Spacetimeformer_Forecaster
    from sys import platform

    if platform == "darwin":
        LOG_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/"
        DATA_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/short_loaded_data_normalized.parquet"
        NORM_CONSTS = pd.read_csv("/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/short_loaded_data_normalization_consts.csv", index_col=None)
        n_workers = mp.cpu_count()
        accelerator = "auto"
        devices = "auto"
        num_nodes = 1
        strategy = "auto"
        model_class = "Spacetimeformer_Forecaster"
    elif platform == "linux":
        LOG_DIR = "/projects/ssc/ahenry/wind_forecasting/logging/"
        DATA_PATH = "/projects/ssc/ahenry/wind_forecasting/awaken_data/filled_data_normalized.parquet"
        NORM_CONSTS = pd.read_csv("/projects/ssc/ahenry/wind_forecasting/awaken_data/filled_data_normalization_consts.csv", index_col=None)
        n_workers = int(os.environ["SLURM_GPUS_ON_NODE"])
        accelerator = "auto"
        devices = 2
        num_nodes = 1
        strategy = "ddp_find_unused_parameters_true"
        model_class = "Spacetimeformer_Forecaster"

    ## DEFINE CONFIGURATION
    config = {
        "experiment": {
            "run_name": "windfarm_debug",
            "log_dir": LOG_DIR
        },
        "dataset": {
            "data_path": DATA_PATH,
            "normalization_consts": NORM_CONSTS,
            "context_length": 9, # 120=10 minutes for 5 sec sample size,
            "prediction_length":  3, # 120=10 minutes for 5 sec sample size,
            # "target_turbine_ids": ["wt029", "wt034", "wt074"],
            "normalize": False, 
            "batch_size": 128,
            "workers": n_workers,
            "overfit": False,
            "test_split": 0.15,
            "val_split": 0.15,
            "collate_fn": None,
            "resample_freq": "30s",
            "n_splits": 2, # how many divisions of each continuity group to make, which is further subdivided into training, test, and validation data
            "dataset_kwargs": { # specific to class KPWindFarm or similar 
                "target_turbine_ids": ["wt001", "wt002", "wt003"]
            }
        },
        "model": {
            "model_class": model_class,
            'embed_size': 32, # Determines dimension of the embedding space
            'num_layers': 3, # Number of transformer blocks stacked
            'num_heads': 4, # Number of heads for spatio-temporal attention
            'forward_expansion': 4, # Multiplier for feedforward network size
            'output_size': 1, # Number of output variables,
            "d_model": 64,
            "d_queries_keys": 64, 
            "d_values": 64, 
            "d_ff": 64
        },
        "callbacks": {
            "progress_bar": {}, 
            "early_stopping": {}, 
            "model_checkpoint": {}, 
            "lr_monitor": {True}
        },
        "trainer": {
            # "grad_clip_norm": 0.0, # Prevents gradient explosion if > 0 
            "limit_val_batches": 1.0, 
            "val_check_interval": 1.0,
            "accelerator": accelerator,
            "devices": devices,
            "num_nodes": num_nodes,
            "strategy": strategy,
            # "debug": False, 
            # "accumulate": 1.0,
            "max_epochs": 1, # Maximum number of epochs to train, 100
            "limit_train_batches": 100
            # "precision": '32-true', # 16-mixed enables mixed precision training, 32-true is full precision
            # 'batch_size': 32, # larger = more stable gradients
            # 'lr': 0.0001, # Step size
            # 'dropout': 0.1, # Regularization parameter (prevents overfitting)
            # 'patience': 50, # Number of epochs to wait before early stopping
            # 'accumulate_grad_batches': 2, # Simulates a larger batch size
        }
    }

    ## SETUP LOGGING

    if not os.path.exists(config["experiment"]["log_dir"]):
        os.makedirs(config["experiment"]["log_dir"])
    wandb_logger = WandbLogger(
        project="wf_forecasting",
        name=config["experiment"]["run_name"],
        log_model=True,
        save_dir=config["experiment"]["log_dir"],
        config=config
    )

    # %%
    ## CREATE DATASET
    # data_module = DataModule(
    #         dataset_class=globals()[config["dataset"]["dataset_class"]],
    #         config=config
    # )
    data_module = DataModule(data_path=DATA_PATH, n_splits=config["dataset"]["n_splits"], train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["dataset_kwargs"]["target_turbine_ids"],
                                one_dim_target=True)

    data_module = DataModule(data_path=DATA_PATH, n_splits=config["dataset"]["n_splits"], train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["dataset_kwargs"]["target_turbine_ids"],
                                one_dim_target=False)
 
        # # dataset = get_dataset("electricity", regenerate=False)
        # from gluonts.multivariate.datasets.dataset import make_multivariate_dataset
        # dataset = make_multivariate_dataset(
        #         dataset_name="electricity_nips",
        #         num_test_dates=7,
        #         prediction_length=24,
        #         max_target_dim=None,
        #     )
        # # n_output_vars = len(list(iter(dataset.train)))
        # n_output_vars = dataset.target_dim
        # train_dataset = dataset.train_ds
        # test_dataset = dataset.test_ds
        # freq = dataset.freq
        # prediction_length = dataset.prediction_length
        # context_length = dataset.prediction_length * 2
        # num_feat_dynamic_real = 0
        # num_feat_static_cat = 1
        # num_feat_static_real = 0

    # %% 
    
    from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
    estimator = InformerEstimator(
        freq=freq, 
        prediction_length=prediction_length,
        context_length=context_length,
        num_feat_dynamic_real=num_feat_dynamic_real, 
        num_feat_static_cat=num_feat_static_cat,
        num_feat_static_real=num_feat_static_real,
        input_size=n_output_vars,
        scaling=False,
        dim_feedforward=config["model"]["d_ff"],
        d_model=config["model"]["d_model"],
        num_encoder_layers=config["model"]["num_layers"],
        num_decoder_layers=config["model"]["num_layers"],
        nhead=config["model"]["num_heads"],
        activation="relu",
        time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
        distr_output=LowRankMultivariateNormalOutput(dim=n_output_vars, rank=8),
        trainer_kwargs={**config["trainer"], "logger": wandb_logger}
    )

    # %%
    if TRAIN:
        predictor = estimator.train(
            training_data=train_dataset,
            validation_data=val_dataset,
            forecast_generator=DistributionForecastGenerator(estimator.distr_output)
            # shuffle_buffer_length=1024
        )
    
    # %%
    # set forecast_generator to DistributionForecastGenerator to access mean and variance in InformerEstimator.create_predictor
    pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    if TEST and os.path.exists(pretrained_filename):
        logging.info("Found pretrained model, loading...")
        model = estimator.create_lightning_module().load_from_checkpoint(pretrained_filename)
        predictor = estimator
        
        forecast_it, ts_it = make_evaluation_predictions(
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
        
        evaluator = MultivariateEvaluator(num_workers=None)

        # %%
        agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts), num_series=data_module.num_target_vars)

        # %%
        agg_df = defaultdict(list)
        # for t, target in enumerate(target_cols):
        for k, v in agg_metrics.items():
            if "_" in k and k.split("_")[0].isdigit():
                target_idx = int(k.split("_")[0])
                turbine_id = target_cols[target_idx].split("_")[-1]
                target_metric = "_".join(target_cols[target_idx].split("_")[:-1])
                perf_metric = k.split('_')[1]
                print(f"Performance metric {perf_metric} for target {target_metric} and turbine {turbine_id} = {v}")
                agg_df["turbine_id"].append(turbine_id)
                agg_df["target_metric"].append(target_metric)
                agg_df["perf_metric"].append(perf_metric)
                agg_df["values"].append(v)

        agg_df = pd.DataFrame(agg_df)
        agg_df = pd.pivot(agg_df, columns="perf_metric") #, values="values", index=["target_metric", "turbine_id"])

        # %%
        forecasts[0].distribution.loc.cpu().numpy()
        forecasts[0].distribution.cov_diag.cpu().numpy()
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
