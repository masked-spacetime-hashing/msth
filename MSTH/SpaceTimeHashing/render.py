import numpy as np
from nerfstudio.cameras.cameras import Cameras
import torch

"""mainly adapted from TensoRF"""


def get_elem(tensor):
    """get element from a tensor assuming tensor contains only one distinct element"""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return float(tensor[0][0].item())


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))  # camera x axis in world coord
    vec1 = normalize(np.cross(vec2, vec0))  # camera y axis in world coord
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=300):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_render_cameras(
    train_cameras: Cameras,
    ref_camera: Cameras,
    near,
    far,
    rads_scale=1.0,
    N_views=300,
    n_frames=300,
    downscale=1,
    offset=[-0.05, 0, 0],
    # offset=[0.0, 0, 0],
):
    dt = 0.75
    c2ws_all = train_cameras.camera_to_worlds.numpy()
    c2w = average_poses(c2ws_all)
    print(c2w)
    up = np.array([0.0, 0.0, 1.0])

    # focal = 0.5
    focal = 1.0
    print("focal", focal)

    zdelta = near * 0.2
    rads = np.array([0.45400773, 0.1343679, 0.05063616]) * rads_scale
    print(rads)
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=0.9, N=n_frames)

    rposes = np.stack(render_poses, axis=0)
    # rposes[..., 0, 3] -= 0.2
    rposes[..., :3, 3] += np.array(offset)
    render_c2ws = torch.from_numpy(rposes)[..., :3, :].to(torch.float32)
    times = torch.linspace(0, n_frames - 1, n_frames) / n_frames
    times = times[..., None].to(torch.float32)

    H = int(get_elem(train_cameras.height))
    W = int(get_elem(train_cameras.width))
    cx = get_elem(train_cameras.cx)
    cy = get_elem(train_cameras.cy)
    print("H", H)
    print("W", W)
    print("cx", cx)
    print("cy", cy)

    render_cams = Cameras(
        render_c2ws,
        fx=get_elem(train_cameras.fx),
        fy=get_elem(train_cameras.fy),
        cx=cx,
        cy=cy,
        width=W,
        height=H,
        times=times,
    )
    render_cams.rescale_output_resolution(1 / downscale)

    return render_cams
