import numpy as np
import jax.numpy as jnp

from jax import grad, jit, vmap, pmap
from jax import lax

from jax import make_jaxpr
from jax import random
from jax import device_put
import matplotlib.pyplot as plt


# Fact 1: jax syntax is similar to numpy
x_np = np.linspace(0, 10, 100)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
# plt.plot(x_np, y_np)
# plt.show()

x_jnp = jnp.linspace(0, 10, 100)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)

print("Shape of x_jnp:", x_jnp.shape)
print("Shape of y_jnp:", y_jnp.shape)
# plt.plot(x_jnp, y_jnp)
# plt.show()



# Fact 2: jax arrays are immutable
size = 10
x = jnp.arange(size)
print(x)

# error out: TypeError: JAX arrays are immutable
# x[0] = 100

# workaround
y = x.at[0].set(100)
print(y)



# Fact 3: JAX handles random numbers differently
seed = 0
key = random.PRNGKey(seed)
x = random.normal(key, (10, ))  # need to explicitly pass the key i.e. PRNG state
print(type(x), x.shape, x.dtype, x)

