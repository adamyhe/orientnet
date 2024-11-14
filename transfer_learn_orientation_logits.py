# Second attempt. Chop off the output layers, add predict w/ linear layer (logits, apply sigmoid + rescaling in loss function).

import json
import math
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tqdm.keras import TqdmCallback

import custom_loss
import ogen
import rnn_v10

fold = int(sys.argv[1])
if len(sys.argv) > 2:
    gpu = int(sys.argv[2])
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[gpu], "GPU")


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


outdir = Path(f"ensemble_models_logits/f{fold}/")

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
pretrained_model = tf.keras.models.load_model(
    f"../clipnet/ensemble_models/fold_{fold}.h5", compile=False
)

# Freeze batch normalization layers
for layer in pretrained_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False


# Define new model
new_output = layers.Dropout(0.3, name="new_dropout")(
    layers.BatchNormalization(name="new_batch_normalization")(
        layers.Dense(1, name="new_dense")(
            layers.GlobalAvgPool1D(name="new_global_avg_pool_1d")(
                pretrained_model.get_layer("max_pooling1d_2").output
            )
        )
    )
)

new_model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=new_output)

# Compile
new_model.compile(
    optimizer=rnn_v10.optimizer(**rnn_v10.opt_hyperparameters),
    loss=custom_loss.rescale_bce,
    metrics={"new_dropout": custom_loss.rescale_corr},
)

# Train model
model_filepath = str(outdir.joinpath("orientnet.h5"))
checkpt = tf.keras.callbacks.ModelCheckpoint(
    model_filepath, verbose=0, save_best_only=True
)
early_stopping = tf.keras.callbacks.EarlyStopping(verbose=1, patience=10)
tqdm_callback = TqdmCallback(verbose=1, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
csv_logger = CSVLogger(
    filename=outdir.joinpath("orientnet.log"), separator=",", append=True
)
fit_model = new_model.fit(
    x=train_gen,
    validation_data=val_gen,
    epochs=rnn_v10.epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=0,
    callbacks=[
        checkpt,
        early_stopping,
        tqdm_callback,
        csv_logger,
        LearningRateScheduler(warmup_lr),
    ],
)
