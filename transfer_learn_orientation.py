import json
import math
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.losses import BinaryCrossentropy
from tqdm.keras import TqdmCallback

import clipnet
import custom_loss
import ogen
import rnn_v10

fold = int(sys.argv[1])
gpu = int(sys.argv[2])


def warmup_lr(epoch, lr):
    """
    Learning rate warmup schedule.
    """
    print(f"LEARNING RATE = {lr}")
    if epoch < 1:
        return lr / 10
    elif epoch == 1:
        return lr * 10
    else:
        return lr


outdir = Path(f"ensemble_models/f{fold}/")

# Create dataset generators
with open(outdir.joinpath("dataset_params.json"), "r") as f:
    dataset_params = json.load(f)
steps_per_epoch = math.floor(
    sum(dataset_params["n_samples_per_train_fold"]) * 2 / rnn_v10.batch_size
)
steps_per_val_epoch = math.floor(
    sum(dataset_params["n_samples_per_val_fold"]) * 2 / rnn_v10.batch_size
)
train_args = [
    dataset_params["train_seq"],
    dataset_params["train_procap"],
    steps_per_epoch,
    rnn_v10.batch_size,
    dataset_params["pad"],
]
val_args = [
    dataset_params["val_seq"],
    dataset_params["val_procap"],
    steps_per_val_epoch,
    rnn_v10.batch_size,
    dataset_params["pad"],
]
train_gen = ogen.OGen(*train_args)
val_gen = ogen.OGen(*val_args)

# Load model
nn = clipnet.CLIPNET(n_gpus=1, use_specific_gpu=gpu)
pretrained_model = tf.keras.models.load_model(
    f"../clipnet/ensemble_models/fold_{fold}.h5", compile=False
)

# Freeze batch normalization layers
for layer in pretrained_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
# Define new output layer
new_output = layers.Activation("sigmoid")(
    layers.BatchNormalization()(
        layers.Dense(1)(
            layers.GlobalAvgPool1D()(
                pretrained_model.get_layer("max_pooling1d_2").output
            )
        )
    )
)
pretrained_model = tf.keras.models.Model(
    inputs=pretrained_model.input,
    outputs=new_output,
)

pretrained_model.compile(
    optimizer=rnn_v10.optimizer(**rnn_v10.opt_hyperparameters),
    loss=BinaryCrossentropy(),
    metrics={"pearson": custom_loss.corr},
)

# Train model
model_filepath = str(outdir.joinpath("orientnet.h5"))
checkpt = tf.keras.callbacks.ModelCheckpoint(
    model_filepath, verbose=0, save_best_only=True
)
early_stopping = tf.keras.callbacks.EarlyStopping(verbose=1, patience=10)
training_time = clipnet.TimeHistory()
tqdm_callback = TqdmCallback(verbose=1, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
csv_logger = CSVLogger(
    filename=outdir.joinpath("transfer.log"),
    separator=",",
    append=True,
)
fit_model = pretrained_model.fit(
    x=train_gen,
    validation_data=val_gen,
    epochs=rnn_v10.epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=0,
    callbacks=[
        checkpt,
        early_stopping,
        training_time,
        tqdm_callback,
        csv_logger,
        LearningRateScheduler(warmup_lr),
    ],
)
