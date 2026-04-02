import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class Encoder(nn.Module):
    repr_dim: int
    network_width: int
    network_depth: int
    skip_connections: int
    use_relu: bool
    use_ln: bool

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        logging.info("encoder input shape: %s", data.shape)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.repr_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    network_width: int
    network_depth: int
    skip_connections: int
    use_relu: bool
    use_ln: bool
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        logging.info("actor input shape: %s", x.shape)
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init, name=f"hidden_{i}")(x)
            if self.use_ln:
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std


class QNetwork(nn.Module):
    """Q-network for SAC (takes obs and action, outputs Q-values for multiple critics).
    
    Each critic has its own complete MLP, matching the original SAC implementation.
    """
    obs_size: int
    action_size: int
    network_width: int
    network_depth: int
    use_relu: bool
    use_ln: bool
    n_critics: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        hidden = jnp.concatenate([obs, actions], axis=-1)
        
        # Each critic gets its own complete MLP (matching original SAC)
        q_values = []
        for i in range(self.n_critics):
            q = hidden
            # Full MLP for this critic
            for j in range(self.network_depth):
                q = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init, name=f"critic_{i}_hidden_{j}")(q)
                if j != self.network_depth - 1:  # Don't normalize/activate final layer
                    if self.use_ln:
                        q = nn.LayerNorm(name=f"critic_{i}_ln_{j}")(q)
                    q = activation(q)
            # Final output layer
            q = nn.Dense(1, kernel_init=lecun_uniform, bias_init=bias_init, name=f"critic_{i}_output")(q)
            q_values.append(q)
        
        return jnp.concatenate(q_values, axis=-1)
