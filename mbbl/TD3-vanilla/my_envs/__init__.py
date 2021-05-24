from gym.envs.registration import registry, register, make, spec


register(
    id='HalfCheetahModified-leg-v12',
    entry_point='my_envs.envs:HalfCheetahEnv_modified_leg',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-base-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedBaseEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-multi-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedMultiEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-mass-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedMassEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-motor-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedMotorEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-jointfriction-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedJointFrictionEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-groundfriction-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedGroundFrictionEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-damping-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedDampingEnv',
    #max_episode_steps=1000,
    #reward_threshold=4800,
)

register(
    id='HalfCheetahModified-stiffness-v12',
    entry_point='my_envs.envs:HalfCheetahModifiedStiffnessEnv',
    max_episode_steps=1000,
    reward_threshold=4800,
)




register(
    id='HalfCheetahModified-base-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedBaseEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-multi-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedMultiEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-mass-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedMassEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-motor-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedMotorEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-jointfriction-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedJointFrictionEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-groundfriction-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedGroundFrictionEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-damping-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedDampingEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)

register(
    id='HalfCheetahModified-stiffness-v13',
    entry_point='my_envs.envs:HalfCheetahModifiedStiffnessEnv131',
    max_episode_steps=1000,
    reward_threshold=4800,
)
