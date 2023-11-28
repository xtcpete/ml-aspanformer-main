import torch
import numpy as np
import math
import pyproj
from scipy.spatial.transform import Rotation
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import cv2
import json

def load_image(filename, RESIZE=None):
    img = cv2.imread(filename)
    H, W, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if RESIZE > 1:
        img = cv2.resize(img, dsize=(W // RESIZE, H // RESIZE))
    return img

def load_depth(filename, RESIZE=None):
    d = np.load(filename)
    H, W = d.shape
    if RESIZE > 1:
        d = cv2.resize(d, dsize=(W // RESIZE, H // RESIZE))
    return d

def load_cam(filename, RESIZE=None):
    with open(filename, 'r') as f:
        cam = json.load(f)
    extr = cam['extrinsics']
    intr = cam['intrinsics']

    lat, lon, alt = extr['lat'], extr['lon'], extr['alt']
    lat_org, lon_org, alt_org = 0,0,0
    omega, phi, kappa = extr['omega'], extr['phi'], extr['kappa']

    trans_mat = geodetic_to_enu(lat, lon, alt, lat_org, lon_org, alt_org)
    rot_mat = opk_to_rotation([omega, phi, kappa])

    transform_mat = np.zeros((4, 4))
    transform_mat[:3, :3] = rot_mat
    transform_mat[:3, 3] = trans_mat
    transform_mat[3, 3] = 1

    K = create_intrinsics(intr)
    if RESIZE > 1:
        K[0, 0] /= RESIZE
        K[1, 1] /= RESIZE
        K[0, 2] /= RESIZE
        K[1, 2] /= RESIZE

    return transform_mat, K

def create_world_coordinates(depth, c2w, K, save_pc=False, save_dir=None, source='carla'):
    H, W = depth.shape

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    homo_uv = np.stack([i, j, np.ones_like(i, dtype=np.float32)], axis=-1)

    if source == 'carla':
        F = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float32)
        F = np.linalg.inv(F)
        cam = ((F @ np.linalg.inv(K)) [None, ...] @ homo_uv.reshape(-1, 3)[..., None])[..., 0]
    elif source == 'AWS':
        cam = (np.linalg.inv(K)[None, ...] @ homo_uv.reshape(-1, 3)[..., None])[..., 0]
    
    cam *= depth.reshape(-1, 1)
    bottom = np.ones_like(cam[..., 0:1])
    homo_cam = np.concatenate([cam, bottom], axis=-1)
    world = (c2w[None, ...] @ homo_cam[..., None])[..., :-1, 0]
    
    if save_pc:
        with open(save_dir, 'w') as f:
            f.write('# OBJ file\n')
            for point in world:
                f.write('v %.4f %.4f %.4f\n' % (point[0], point[1], point[2]))
        f.close()

    return world.reshape(H, W, -1)

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    bottom = np.ones_like(loc[..., 0:1], dtype=loc.dtype)
    point = np.concatenate([loc, bottom], axis=-1)
        # transform to camera coordinates
    # point_camera = np.dot(w2c, point)
    point_camera = (w2c[None, None, ...] @ point[..., None])[..., 0]
    
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = np.stack([point_camera[..., 1], -point_camera[..., 2], point_camera[..., 0]], axis=-1)
    # now project 3D->2D using the camera matrix
    # point_img = np.dot(K, point_camera)
    point_img = (K[None, None, ...] @ point_camera[...,:3, None])[..., 0]
    # normalize
    point_img[..., 0] /= point_img[..., 2]
    point_img[..., 1] /= point_img[..., 2]

    return point_img[..., :2]

def get_warpped_image(depth, K, rgb, point2D):
    W, H = int(K[0, 2] * 2), int(K[1, 2] * 2)
    warpped_image = np.zeros([H, W, 3], dtype=rgb.dtype)

    pixel = np.argwhere(depth+1)[:, [1, 0]]
    pixel = pixel.reshape(H, W, 2)

    for i in range(H):
        for j in range(W):
            ori_x, ori_y = int(pixel[i, j, 0]), int(pixel[i, j, 1])
            new_x, new_y = int(point2D[i, j, 0]), int(point2D[i, j, 1])
            if new_x < 0 or new_x >= W or new_y < 0 or new_y >= H:
                continue
            warpped_image[new_y, new_x, :] = rgb[ori_y, ori_x, :]
    return warpped_image

def fov2focal(fov, H, W):
    fx = W / (2 * np.tan(fov / 2 / 180 * np.pi))
    fy = H / (2 * np.tan(fov / 2 / 180 * np.pi))

    return fx, fy

def geodetic_to_enu(lat, lon, alt, lat_org, lon_org, alt_org):
    """
    convert LLA to ENU
    :params lat, lon, alt: input LLA coordinates
    :params lat_org, lon_org, alt_org: LLA of the origin of the local ENU coordinate system
    :return: east, north, up coordinate
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    x_org, y_org, z_org = transformer.transform(
        lon_org, lat_org, alt_org, radians=False
    )
    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T
    rot1 = Rotation.from_euler(
        "x", -(90 - lat_org), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1
    rot3 = Rotation.from_euler(
        "z", -(90 + lon_org), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1
    rotMatrix = rot1.dot(rot3)
    enu = rotMatrix.dot(vec).T.ravel()
    return enu.T

def opk_to_rotation(opk_degrees):
    """
    calculate a rotation matrix given euler angles
    :params opk: list of [omega, phi, kappa] angles (degrees)
    :return: rotation matrix
    """
    opk = np.radians(opk_degrees)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(opk[0]), -math.sin(opk[0])],
            [0, math.sin(opk[0]), math.cos(opk[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(opk[1]), 0, math.sin(opk[1])],
            [0, 1, 0],
            [-math.sin(opk[1]), 0, math.cos(opk[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(opk[2]), -math.sin(opk[2]), 0],
            [math.sin(opk[2]), math.cos(opk[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R


def get_utm_convergence_matrix(lat, lon, alt):
    """
    compute convergence matrix from true North for UTM projection to WGS84
    :params lat,lon,alt: input WGS84 coordinates
    :return: convergence matrix to convert UTM grid north to true north angle (degrees)
    """
    delta = 1e-6
    p1 = np.array(lla_to_utm(lat + delta, lon, alt))
    p2 = np.array(lla_to_utm(lat, lon, alt))
    xnp = p1 - p2
    angle = math.atan2(xnp[0], xnp[1])
    R_c = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return R_c

def lla_to_utm(lat, lon, alt):
    """
    convert WGS84 coordinate to UTM
    :param lat,lon,alt: WGS84 coordinate
    :return: UTM x,y,z
    """
    utm_epsg = utm_epsg_from_wgs84(lat, lon)
    transformer = pyproj.Transformer.from_crs(4326, utm_epsg, always_xy=True)
    x, y, z = transformer.transform(lon, lat, alt)
    return x, y, z

def utm_epsg_from_wgs84(latitude, longitude):
    """
    get UTM EPSG code for input WGS84 coordinate
    :param latitude, longitude: WGS84 coordinate in degrees
    :return: UTM EPSG code
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude,
            south_lat_degree=latitude,
            east_lon_degree=longitude,
            north_lat_degree=latitude,
        ),
    )
    utm_epsg = int(utm_crs_list[0].code)
    return utm_epsg


def utm_opk_to_enu_opk(lat, lon, alt, omega_utm, phi_utm, kappa_utm):
    """
    convert UTM OPK angles to ENU OPK angles
    :params lat, lon, alt: camera location in WGS84 coordinates
    :params omega_utm, phi_utm, kappa_utm: camera OPK angles in UTM projection
    :return omega_enu, phi_enu, kappa_enu: camera OPK angles in ENU
    """
    # build rotation matrix from OPK angles
    R = opk_to_rotation([omega_utm, phi_utm, kappa_utm])
    # get UTM convergence angle rotation matrix and map UTM OPK to ENU
    R_c = get_utm_convergence_matrix(lat, lon, alt)
    R = np.dot(R_c, R)
    omega_enu, phi_enu, kappa_enu = rotation_to_opk(R)
    return omega_enu, phi_enu, kappa_enu

def rotation_to_opk(R):
    """
    calculate euler angles from a rotation matrix
    :param R: input rotation matrix
    :return opk: list of [omega, phi, kappa] angles (degrees)
    """
    omega = np.degrees(np.arctan2(-R[1][2], R[2][2]))
    phi = np.degrees(np.arcsin(R[0][2]))
    kappa = np.degrees(np.arctan2(-R[0][1], R[0][0]))
    return [omega, phi, kappa]



def create_camera_pose(extrinsic, source='AWS'):
    cam_pos = geodetic_to_enu(extrinsic['lat'], extrinsic['lon'], extrinsic['alt'], 0, 0, 0)

    R = opk_to_rotation([extrinsic['omega'], extrinsic['phi'], extrinsic['kappa']])

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = cam_pos
    pose[3, 3] = 1
    pose = fix_pose(pose) if source == 'AWS' else pose

    return pose

def fix_pose(pose):
    # 3D Rotation about the x-axis.
    t = np.pi
    c = np.cos(t)
    s = np.sin(t)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    axis_transform = np.eye(4)
    axis_transform[:3, :3] = R
    return pose @ axis_transform


def create_intrinsics(intrinsic, source='AWS'):
    W = intrinsic['columns'] if 'columns' in intrinsic else intrinsic['width']
    H = intrinsic['rows'] if 'rows' in intrinsic else intrinsic['height']
    if 'fov' in intrinsic:
        f = W / 2 / np.tan(intrinsic['fov'] / 2 / 180 * np.pi)
        fx = f
        fy = f
        cx = W / 2
        cy = H / 2
    else:
        fx = intrinsic['fx']
        fy = intrinsic['fy']
        cx = intrinsic['cx']
        cy = intrinsic['cy']
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    F = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float32)
    F = np.linalg.inv(F)

    K = K if source == 'AWS' else np.linalg.inv(F @ np.linalg.inv(K))

    return K

def create_world_coordinates_np(depth, cam2world_matrix, intrinsics, save_pc=False, save_dir=None):
    H, W = depth.shape

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # [H, W, 2]

    homo_uv = np.stack([i, j, np.ones_like(i, dtype=np.float32)], axis=-1) # [H, W, 1]
    # F = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float32)
    # F = np.linalg.inv(F)

    # get 3D coords
    cam = (np.linalg.inv(intrinsics)[None, ...] @ homo_uv.reshape(-1, 3)[..., None])[..., 0] # [HW, 3]
    # cam = ((F @ np.linalg.inv(intrinsics))[None, ...] @ homo_uv.reshape(-1, 3)[..., None])[..., 0]
    cam *= depth.reshape(-1, 1) # [HW, 3]

    
    bottom = np.ones_like(cam[..., 0:1]) #[HW, 1]
    homo_cam = np.concatenate([cam, bottom], axis=-1) #[HW, 4]
    world = (cam2world_matrix[None, ...] @ homo_cam[..., None])[..., :-1, 0] # [1, 4, 4] @ [HW, 4, 1] -> [HW, 4, 1] -> [HW, 3]
    
    if save_pc:
        with open(save_dir, 'w') as f:
            f.write('# OBJ file\n')
            for point in world:
                f.write('v %.4f %.4f %.4f\n' % (point[0], point[1], point[2]))
        f.close()

    return world.reshape(H, W, -1) # [H, W, 3]
    

if __name__ == "__main__":
    uv = torch.stack(
            torch.meshgrid(
                torch.arange(10, dtype=torch.float32),
                torch.arange(6, dtype=torch.float32),
                indexing='ij'
            )
        )
    i, j = torch.meshgrid(torch.arange(6, dtype=torch.float32), torch.arange(10, dtype=torch.float32), indexing='ij')
    # print(uv.shape)
    # print(uv[:, 0, 1])
    # print(uv[:, 1, 0])
    print(i.shape)
    print(j.shape)