import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar


@jax.jit
def cross_entropy_loss_and_accuracy(
    logits: Float[Array, " seq_length vocab_size"],
    tokens: Int[Array, " seq_length"],
    valid=None,
) -> tuple[Scalar, Scalar]:
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)

    log_prob = jax.nn.log_softmax(logits, axis=-1)
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(log_prob, jnp.expand_dims(tokens, -1), axis=-1),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))

    token_wise_loss = -token_log_prob
    loss_pure_ce = jnp.mean(jnp.sum(token_wise_loss, axis=-1) / valid_text_length)
    loss = jnp.mean(jnp.sum(token_wise_loss, axis=-1) / valid_text_length)

    return loss, loss_pure_ce


def token_log_probs(logits, targets) -> jnp.ndarray:
    token_log_probs = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(targets, -1),
            axis=-1,
        ),
        -1,
    )
    return token_log_probs
