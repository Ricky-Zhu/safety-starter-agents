from tensorflow.python.util import deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from safe_rl.pg.algos import ppo, ppo_lagrangian, trpo, trpo_lagrangian, cpo, cpo_within_BO, \
    trpo_lagrangian_within_BO, ppo_lagrangian_within_BO, saute_ppo_lagrangian_whin_BO, saute_ppo_lagrangian,saute_trpo
from safe_rl.sac.sac import sac
