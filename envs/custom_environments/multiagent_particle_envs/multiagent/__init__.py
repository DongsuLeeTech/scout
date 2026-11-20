from gym.envs.registration import register

register(
    id="MultiagentSimple-v0",
    entry_point="multiagent.envs:SimpleEnv",
    max_episode_steps=100,
)

register(
    id="MultiagentSimpleSpeakerListener-v0",
    entry_point="multiagent.envs:SimpleSpeakerListenerEnv",
    max_episode_steps=100,
)
