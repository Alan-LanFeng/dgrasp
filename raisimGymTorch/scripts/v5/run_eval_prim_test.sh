#!/bin/sh
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_reg.yaml' -sd 'data_v5'         -w '2021-11-09-15-50-55/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 2 -e '003_crackerbox_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'    -w '2021-11-09-21-24-53/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_reg.yaml' -sd 'data_v5'    -w '2021-11-09-15-49-15/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 4 -e '005_tomatocan_rgpos_015_extstate_v4_3' -c 'cfg_reg.yaml' -sd 'data_v5'  -w '2021-11-09-20-15-24/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 5 -e '006_mustard_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'      -w '2021-11-10-04-16-23/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_reg.yaml' -sd 'data_v5'       -w '2021-11-11-00-01-12/full_2400.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 7 -e '008_pudding_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'       -w '2021-11-09-21-24-53/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 8 -e '009_gelatin_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'       -w '2021-11-09-21-24-54/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 9 -e '010_pottedcan_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-24-52/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 10 -e '011_banana_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'       -w '2021-11-09-21-25-16/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_reg.yaml'  -sd 'data_v5'     -w '2021-11-09-21-24-54/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 12 -e '021_bleach_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'       -w '2021-11-09-21-24-56/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 13 -e '024_bowl_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'         -w '2021-11-09-21-25-16/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 14 -e '025_mug_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'          -w '2021-11-09-21-24-53/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 15 -e '035_powerdrill_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'  -w '2021-11-10-04-16-51/full_3000.pt'   -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 16 -e '036_woodblock_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'    -w '2021-11-10-04-17-38/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 17 -e '037_scissors_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-24-51/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 18 -e '040_marker_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'       -w '2021-11-09-21-24-53/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'           -w '2021-11-09-21-25-25/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_reg.yaml' -sd 'data_v5'         -w '2021-11-11-00-01-23/full_2800.pt' -p

#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_reg.yaml' -sd 'data_v5'         -w '2021-11-09-15-50-55/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 2 -e '003_crackerbox_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'  -w '2021-11-09-21-24-53/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_reg.yaml' -sd 'data_v5'    -w '2021-11-09-15-49-15/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 4 -e '005_tomatocan_rgpos_015_extstate_v4_3' -c 'cfg_reg.yaml' -sd 'data_v5'   -w '2021-11-09-21-25-21/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 5 -e '006_mustard_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-10-04-16-23/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-24-57/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 7 -e '008_pudding_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-24-53/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 8 -e '009_gelatin_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-24-54/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 9 -e '010_pottedcan_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'   -w '2021-11-09-21-24-52/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 10 -e '011_banana_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-25-16/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_reg.yaml'  -sd 'data_v5'   -w '2021-11-09-21-24-54/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 12 -e '021_bleach_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-24-56/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 13 -e '024_bowl_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'       -w '2021-11-09-21-25-16/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 14 -e '025_mug_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'        -w '2021-11-09-21-24-53/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 15 -e '035_powerdrill_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5' -w '2021-11-10-04-16-51/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 16 -e '036_woodblock_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'  -w '2021-11-10-04-17-38/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 17 -e '037_scissors_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'   -w '2021-11-09-21-24-51/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 18 -e '040_marker_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'     -w '2021-11-09-21-24-53/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_reg.yaml' -sd 'data_v5'      -w '2021-11-09-21-25-25/full_3000.pt' -t -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_reg.yaml' -sd 'data_v5'    -w '2021-11-11-00-01-23/full_3000.pt' -t -p