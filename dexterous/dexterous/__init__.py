from gym.envs.registration import register

register(
    id="Dexterous-v1",
    entry_point="dexterous.envs:DexterousRealArmEnv",
    max_episode_steps=400,
)
