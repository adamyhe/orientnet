"""This implements a number of custom loss/metric functions for use in CLIPNET."""

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


def pearsonr(y_true, y_pred):
    """Deprecated. Use correlation_loss."""
    true_residual = y_true - tf.math.mean(y_true)
    pred_residual = y_pred - tf.math.mean(y_pred)
    num = tf.math.sum(tf.math.multiply(true_residual, pred_residual))
    den = tf.math.sqrt(
        tf.math.multiply(
            tf.math.sum(tf.math.square(true_residual)),
            tf.math.sum(tf.math.square(pred_residual)),
        )
    )
    r = num / den
    return r  # makes function decreasing and non-zero


def corr(x, y, pseudocount=1e-6):
    """
    Computes Pearson's r between x and y. Pseudocount ensures non-zero denominator.
    """
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    num = tf.math.reduce_mean(tf.multiply(xm, ym))
    den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym) + pseudocount
    r = tf.math.maximum(tf.math.minimum(num / den, 1), -1)
    return r


def corr_loss(x, y, pseudocount=1e-6):
    """Computes -correlation(x, y)."""
    return -corr(x, y, pseudocount)


def squared_log_sum_error(x, y, pseudocount=1e-6):
    """
    Computes the squared difference between log sums of vectors. Pseudocount ensures
    non-zero log inputs.
    """
    log_sum_x = tf.math.log(tf.math.reduce_sum(x) + pseudocount)
    log_sum_y = tf.math.log(tf.math.reduce_sum(y) + pseudocount)
    return (log_sum_x - log_sum_y) ** 2


def cosine_slse(x, y, slse_scale=8e-3, pseudocount=1e-6):
    """Computes cosine loss + scale * slse."""
    cosine_loss = tf.keras.losses.CosineSimilarity()
    return cosine_loss(x, y).numpy() + slse_scale * squared_log_sum_error(
        x, y, pseudocount
    )


def sum_error(x, y):
    return tf.math.reduce_sum(x) - tf.math.reduce_sum(y)


def sum_true(x, y):
    return tf.math.reduce_sum(x)


def sum_pred(x, y):
    return tf.math.reduce_sum(y)


def jaccard_distance(y_true, y_pred, smooth=100):
    """Calculates mean of Jaccard distance as a loss function"""
    y = tf.cast(y_true, tf.float32)
    intersection = tf.math.reduce_sum(y * y_pred)
    sum_ = tf.math.reduce_sum(y + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return jd  # tf.reduce_mean(jd)


def multinomial_nll(true_counts, logits):
    """
    Compute the multinomial negative log-likelihood along the sequence (axis=1)
    and sum the values across all channels

    Adapted from Avsec et al. (2021) Nature Genetics. https://doi.org/10.1038/s41588-021-00782-6
    Args:
      true_counts: observed count values (batch, seqlen, channels)
      logits: predicted logit values (batch, seqlen, channels)
    """
    # round sum to nearest int
    counts_per_example = tf.math.round(tf.reduce_sum(true_counts, axis=-1))
    # compute distribution
    dist = tf.compat.v1.distributions.Multinomial(
        total_count=counts_per_example, logits=logits
    )
    # return negative log probabilities
    return -tf.reduce_sum(dist.log_prob(true_counts))


def beta_regression_loss(y_true, y_pred):
    """
    Beta regression loss function for Keras models.
    """
    # Ensure the model outputs (alpha and beta) are positive
    alpha = tf.exp(y_pred[..., 0])  # Exponentiate to ensure positivity
    beta = tf.exp(y_pred[..., 1])  # Exponentiate to ensure positivity

    # Log likelihood of the Beta distribution
    log_likelihood = (alpha - 1) * tf.math.log(y_true) + (beta - 1) * tf.math.log(
        1 - y_true
    )
    log_beta_normalization = (
        tf.math.lgamma(alpha) + tf.math.lgamma(beta) - tf.math.lgamma(alpha + beta)
    )

    # Negative log-likelihood (for minimization)
    negative_log_likelihood = -tf.reduce_mean(log_likelihood - log_beta_normalization)
    return negative_log_likelihood


def rescale_sigmoid(logits, lower=0.5, upper=1.0):
    scale = upper - lower
    scaled = lower + scale * tf.math.sigmoid(logits)
    return scaled


def rescale(x, lower=0.0, upper=1.0):
    scale = upper - lower
    scaled = lower + scale * x
    return scaled


def rescale_bce(y_true, y_pred, lower=0.5, upper=1.0):
    return BinaryCrossentropy()(y_true, rescale_sigmoid(y_pred, lower, upper))


def rescale_corr(y_true, y_pred, lower=0.5, upper=1.0):
    return corr(y_true, rescale_sigmoid(y_pred, lower, upper))


def rescale_true_bce(y_true, y_pred, lower=0.0, upper=1.0):
    return BinaryCrossentropy()(rescale(y_true, lower, upper), tf.math.sigmoid(y_pred))


def rescale_true_corr(y_true, y_pred, lower=0.0, upper=1.0):
    return corr(rescale(y_true, lower, upper), tf.math.sigmoid(y_pred))
