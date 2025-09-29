__version__ = "0.0.1"
from gymnasium.envs.registration import register


register(
    id="R1ProEnv-v0",
    entry_point="brs_ctrl.env:R1ProEnv",
)
