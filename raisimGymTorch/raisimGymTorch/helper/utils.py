import numpy as np
from raisimGymTorch.helper import rotations
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import random
import os

def concat_dict(dict):
    ret = {}
    for key in dict.keys():
        for k in dict[key].keys():
            if k in ret.keys():
                ret[k] = np.concatenate((ret[k], dict[key][k]), axis=0)
            else:
                ret[k] = dict[key][k]
    return ret

IDX_TO_OBJ = {
    1: ['002_master_chef_can',0.414, 0, [0.051,0.139,0.0]],
    2: ['003_cracker_box', 0.453, 1, [0.06, 0.158, 0.21]],
    3: ['004_sugar_box', 0.514, 1, [0.038, 0.089, 0.175]],
    4: ['005_tomato_soup_can', 0.349, 0, [0.033, 0.101,0.0]],
    5: ['006_mustard_bottle', 0.431,2, [0.0,0.0,0.0]],
    6: ['007_tuna_fish_can', 0.171, 0, [0.0425, 0.033,0.0]],
    7: ['008_pudding_box', 0.187, 3, [0.21, 0.089, 0.035]],
    8: ['009_gelatin_box', 0.097, 3, [0.028, 0.085, 0.073]],
    9: ['010_potted_meat_can', 0.37, 3, [0.05, 0.097, 0.089]],
    10: ['011_banana', 0.066,2, [0.028, 0.085, 0.073]],
    11: ['019_pitcher_base', 0.178,2, [0.0,0.0,0.0]],
    12: ['021_bleach_cleanser', 0.302,2, [0.0,0.0,0.0]], # not sure about weight here
    13: ['024_bowl', 0.147,2, [0.0,0.0,0.0]],
    14: ['025_mug', 0.118,2, [0.0,0.0,0.0]],
    15: ['035_power_drill', 0.895,2, [0.0,0.0,0.0]],
    16: ['036_wood_block', 0.729, 3, [0.085, 0.085, 0.2]],
    17: ['037_scissors', 0.082,2, [0.0,0.0,0.0]],
    18: ['040_large_marker', 0.01, 3, [0.009,0.121,0.0]],
    19: ['051_large_clamp', 0.125,2, [0.0,0.0,0.0]],
    20: ['052_extra_large_clamp', 0.102,2, [0.0,0.0,0.0]],
    21: ['061_foam_brick', 0.028, 1, [0.05, 0.075, 0.05]],
}

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_obj_pcd(path,num_p=100):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    mesh.remove_duplicated_vertices()
    obj_vertices = mesh.sample_points_uniformly(number_of_points=num_p)
    obj_vertices = np.asarray(obj_vertices.points)

    return obj_vertices

def first_nonzero(arr, axis, invalid_val=-1):
    arr = torch.Tensor(arr)
    mask = arr!=0
    mask = mask.to(torch.uint8)
    return torch.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def dgrasp_to_mano(param):
    bs = param.shape[0]
    eulers = param[:,6:].reshape(bs,-1, 3).copy()

    # exchange ring finger and little finger's sequence
    temp = eulers[:,6:9].copy()
    eulers[:,6:9] = eulers[:,9:12]
    eulers[:,9:12] = temp

    eulers = eulers.reshape(-1,3)
    # change euler angle to axis angle
    rotvec = R.from_euler('XYZ', eulers, degrees=False)
    rotvec = rotvec.as_rotvec().reshape(bs,-1)
    global_orient = R.from_euler('XYZ', param[:,3:6], degrees=False)
    global_orient = global_orient.as_rotvec()

    # translation minus a offset
    offset = np.array([[0.09566993, 0.00638343, 0.00618631]])
    mano_param = np.concatenate([global_orient, rotvec, param[:,:3] - offset],axis=1)

    return mano_param

def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out
def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix
def show_pointcloud_objhand(hand, obj):
    '''
    Draw hand and obj xyz at the same time
    :param hand: [778, 3]
    :param obj: [3000, 3]
    '''


    hand_dim = hand.shape[0]
    obj_dim = obj.shape[0]
    handObj = np.vstack((hand, obj))
    c_hand, c_obj = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_hand = np.repeat(c_hand, repeats=hand_dim, axis=0) # [778,3]
    c_obj = np.repeat(c_obj, repeats=obj_dim, axis=0) # [3000,3]
    c_hanObj = np.vstack((c_hand, c_obj)) # [778+3000, 3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(handObj)
    pc.colors = o3d.utility.Vector3dVector(c_hanObj)
    o3d.visualization.draw_geometries([pc])

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg.yaml')
    parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
    parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
    parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
    parser.add_argument('-w', '--weight', type=str, default='full_400.pt')
    parser.add_argument('-sd', '--storedir', type=str, default='data_all')
    parser.add_argument('-pr', '--prior', action="store_true")
    parser.add_argument('-o', '--obj_id', type=int, default=7)
    parser.add_argument('-t', '--test', action="store_true")
    parser.add_argument('-mc', '--mesh_collision', action="store_true")
    parser.add_argument('-ao', '--all_objects', action="store_true")
    parser.add_argument('-to', '--test_object_set', type=int, default=-1)
    parser.add_argument('-ac', '--all_contact', action="store_true")
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-itr', '--num_iterations', type=int, default=3001)
    parser.add_argument('-nr', '--num_repeats', type=int, default=10)
    parser.add_argument('-ev', '--vis_evaluate', action="store_true")
    parser.add_argument('-sv', '--store_video', action="store_true")

    args = parser.parse_args()

    return args

def repeat_label(label_dict, num_repeats):
    ret = {}
    for k,v in label_dict.items():
        if 'type' in k or 'idx' in k or 'name' in k:
            ret[k] = np.repeat(v, num_repeats, 0)
        else :
            ret[k] = np.repeat(v,num_repeats,0).astype('float32')

    return ret

def euler_noise_to_quat(quats, palm_pose, noise):
    eulers_palm_mats = np.array([rotations.euler2mat(pose) for pose in palm_pose]).copy()
    eulers_mats =  np.array([rotations.quat2mat(quat) for quat in quats])

    rotmats_list = np.array([rotations.euler2mat(noise) for noise in noise])

    eulers_new = np.matmul(rotmats_list,eulers_mats)
    eulers_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_new])

    eulers_palm_new = np.matmul(rotmats_list,eulers_palm_mats)
    eulers_palm_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_palm_new])

    quat_list = [rotations.euler2quat(noise) for noise in eulers_rotmated]

    return np.array(quat_list), eulers_new, eulers_palm_rotmated