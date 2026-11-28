"""Repair locally inconsistent low-opacity samples in predominantly opaque GLB materials."""
import numpy as np


def recover_opaque_samples(channels, covered, positions, face_ids, normals, voxel_size):
    """Copy a coherent nearby PBR sample, using UV, surface distance and orientation."""
    rgb, alpha = channels[0], channels[-1].reshape(covered.shape)
    trusted = covered & (alpha >= 245)
    if not covered.any() or trusted.sum() < 0.90 * covered.sum():
        return 0
    from scipy.spatial import cKDTree

    ys, xs = np.nonzero(covered & (alpha < 230))
    if not len(ys):
        return 0
    # A packed UV island can be smaller than a pixel. Search the 3D surface
    # so nearby donors remain reachable even when their atlas charts are far apart.
    dy, dx = np.nonzero(trusted[::2, ::2])
    dy, dx = dy * 2, dx * 2
    if len(dy) < 3:
        return 0
    tree = cKDTree(positions[dy, dx])
    repaired = 0
    for start in range(0, len(ys), 16384):
        y, x = ys[start:start + 16384], xs[start:start + 16384]
        distances, indices = tree.query(positions[y, x], k=32, distance_upper_bound=12 * voxel_size)
        valid = np.isfinite(distances)
        indices = np.minimum(indices, len(dy) - 1)
        yy, xx = dy[indices], dx[indices]
        normal = normals[face_ids[y, x]]
        valid &= (normals[face_ids[yy, xx]] * normal[:, None]).sum(-1) > 0.5
        choice = np.where(valid, distances, np.inf).argmin(1)
        row = np.arange(len(y))
        colors = rgb[yy, xx].astype(np.int16)
        donor = colors[row, choice]
        agree = valid & (np.abs(colors - donor[:, None]).max(-1) <= 28)
        fix = (agree.sum(1) >= 3) & (agree.sum(1) * 4 >= valid.sum(1) * 3)
        fix &= (donor - rgb[y, x]).mean(1) > 12
        for channel in channels:
            channel[y[fix], x[fix]] = channel[yy[row, choice][fix], xx[row, choice][fix]]
        repaired += int(fix.sum())
    return repaired


def pad_material_edges(channels, covered):
    """Extend actual edge colors into the bilinear sampling footprint."""
    import cv2

    if not covered.any() or covered.all():
        return
    distance, labels = cv2.distanceTransformWithLabels(
        (~covered).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    gutter = (~covered) & (distance <= 4)
    source = np.empty(int(labels.max()) + 1, dtype=np.int32)
    source[labels[covered]] = np.flatnonzero(covered)
    y, x = np.unravel_index(source[labels[gutter]], covered.shape)
    for channel in channels:
        channel[gutter] = channel[y, x]
