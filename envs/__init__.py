from gym.envs.registration import register


register(
    id='GoalGrid-v0',
    entry_point='envs.continuous_grid:GoalContinuousGrid',
)

register(
    id='SawyerReachXYZEnv-v1',
    entry_point='envs.sawyer_reach:SawyerReachXYZEnv',
    kwargs={
        'hide_goal_markers': True,
        'norm_order': 2,
    },
)
register(
    id='Point2DLargeEnv-v1',
    entry_point='envs.point2d:Point2DEnv',
    kwargs={
        'images_are_rgb': True,
        'target_radius': 1,
        'ball_radius': 0.5,
        'boundary_dist':5,
        'render_onscreen': False,
        'show_goal': True,
        'render_size':512,
        'get_image_base_render_size': (48, 48),
        'bg_color': 'white',
    },
)
register(
    id='PointFixedEnv-v1',
    entry_point='envs.point2d:Point2DEnv',
    kwargs={
        'images_are_rgb': True,
        'target_radius': 0.2,
        'ball_radius': 0.5,
        'boundary_dist':5,
        'render_onscreen': False,
        'show_goal': True,
        'render_size':512,
        'get_image_base_render_size': (48, 48),
        'bg_color': 'white',
        'fixed_goal_set':True,
        'fixed_init_position': (0,0),
        'randomize_position_on_reset': False
    },
)
register(
    id='Point2D-FourRoom-v1',
    entry_point='envs.point2d:Point2DWallEnv',
    kwargs={
        'action_scale': 1,
        'wall_shape': 'four-room-v1', 
        'wall_thickness': 0.30,
        'target_radius':1,
        'ball_radius':0.5,
        'boundary_dist':5,
        'render_size': 512,
        'wall_color': 'darkgray',
        'bg_color': 'white',
        'images_are_rgb': True,
        'render_onscreen': False,
        'show_goal': True,
        'get_image_base_render_size': (48, 48),
    },
)
# register gcsl envs
register(
    id='SawyerDoor-v0',
    entry_point='envs.sawyer_door:SawyerDoorGoalEnv',
)

register(
    id='DClawTurn-v0',
    entry_point='envs.claw_env:ClawEnv',
    max_episode_steps=80)

