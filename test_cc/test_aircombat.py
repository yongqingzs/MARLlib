import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marllib import marl
RUN_OR_RENDER = 0

def air_mappo():
    # competitive mode
    env = marl.make_env(environment_name="aircombat", map_name="MultipleCombat_2v2/NoWeapon/Selfplay")

    mappo = marl.algos.mappo(hyperparam_source='common')

    # build agent model based on env + algorithms + user preference
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

    # start training
    mappo.fit(env, model, stop={'timesteps_total': 1000}, share_policy='group')

def render_air_mappo():
    # competitive mode
    env = marl.make_env(environment_name="aircombat", map_name="MultipleCombat_2v2/NoWeapon/Selfplay")

    mappo = marl.algos.mappo(hyperparam_source='common')

    # build agent model based on env + algorithms + user preference
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

    params_path_1 = "/params.json"
    model_path_1 = "/checkpoint-1"

    #TODO: change 
    params_path_tmp = "/home/yongqingzs/codes/py/MARLlib/test_cc/exp_results/mappo_mlp_MultipleCombat_2v2/NoWeapon/Selfplay/MAPPOTrainer_aircombat_MultipleCombat_2v2_NoWeapon_Selfplay_6d440_00000_0_2024-10-24_15-56-08"
    model_path_tmp = "/home/yongqingzs/codes/py/MARLlib/test_cc/exp_results/mappo_mlp_MultipleCombat_2v2/NoWeapon/Selfplay/MAPPOTrainer_aircombat_MultipleCombat_2v2_NoWeapon_Selfplay_6d440_00000_0_2024-10-24_15-56-08/checkpoint_000001"

    params_path = params_path_tmp + params_path_1
    model_path = model_path_tmp + model_path_1

    # rendering
    mappo.render(
    env, model, 
    local_mode=True, 
    restore_path={'params_path': params_path,
                    'model_path': model_path},
    lr=0
    )

#maddpg不适用aircombat，因为aircombat是离散动作空间

#maa2c
def air_maa2c():
    env = marl.make_env(environment_name="aircombat", map_name="MultipleCombat_2v2/NoWeapon/Selfplay")

    maa2c = marl.algos.maa2c(hyperparam_source='common')

    model = marl.build_model(env, maa2c, {"core_arch": "mlp", "encode_layer": "128-256"})

    maa2c.fit(env, model, stop={'timesteps_total': 1000000}, share_policy='group')

if __name__ == "__main__":
    if RUN_OR_RENDER:
        air_mappo()
        # air_maa2c()
    else:
        render_air_mappo()
