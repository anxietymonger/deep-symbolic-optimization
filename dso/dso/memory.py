from collections import namedtuple


Batch = namedtuple(
    "Batch", ["actions", "obs", "priors", "lengths", "rewards", "on_policy"])
