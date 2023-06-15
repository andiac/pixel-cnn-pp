import numpy as np
from warp_image import warp_images

def _get_regular_grid(image, points_per_dim):
    nrows, ncols = image.shape[0], image.shape[1]
    rows = np.linspace(0, nrows, points_per_dim)
    cols = np.linspace(0, ncols, points_per_dim)
    rows, cols = np.meshgrid(rows, cols)
    return np.dstack([cols.flat, rows.flat])[0]


def _generate_random_vectors(image, src_points, scale):
    dst_pts = src_points + np.random.uniform(-scale, scale, src_points.shape)
    return dst_pts


def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=False):
    width, height = image.shape[:2]
    if keep_corners:
        corner_points = np.array(
            [[0, 0], [0, width], [height, 0], [height, width]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    out = warp_images(src_points, dst_points,
                      np.moveaxis(image, 2, 0),
                      (0, 0, width, height))
                      # Andi: fix bug, should not be (0, 0, width - 1, height - 1)
                      # (0, 0, width - 1, height - 1))
    return np.moveaxis(np.array(out), 0, 2)


def tps_warp(image, points_per_dim, scale):
    image = np.array(image)
    if len(image.shape) == 2:  # grayscale
        image = np.expand_dims(image, 2)
    width, height = image.shape[:2]
    src = _get_regular_grid(image, points_per_dim=points_per_dim)
    dst = _generate_random_vectors(image, src, scale=scale*width)
    out = _thin_plate_spline_warp(image, src, dst)
    # print(out.shape)
    if len(image.shape) == 2:  # grayscale
        return out.squeeze(2)
    else:
        return out

def tps_warp_2(image, dst, src):
    out = _thin_plate_spline_warp(image, src, dst)
    return out

class TPSWarp:
    def __init__(self, points_per_dim, scale):
        self.points_per_dim = points_per_dim
        self.scale = scale

    def __call__(self, image):
        return tps_warp(image, self.points_per_dim, self.scale)

