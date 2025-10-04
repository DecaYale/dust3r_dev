import numpy as np



def align_point_sets(A, B):
    """
    Aligns two sets of 3D points A and B by finding the optimal scale, rotation, and translation.
    A and B should be arrays of shape (N, 3) where N is the number of points.
    """
    
    # Step 1: Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Step 2: Center the points to the origin
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Step 3: Compute scale (optional, here assuming similar scale)
    # Uncomment the following two lines to compute a scale factor
    # scale = np.linalg.norm(B_centered) / np.linalg.norm(A_centered)
    # A_centered *= scale

    # Step 4: Compute the rotation using Singular Value Decomposition (SVD)
    H = np.dot(A_centered.T, B_centered)
    U, S, Vt = np.linalg.svd(H)
    R_optimal = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R_optimal) < 0:
        Vt[-1, :] *= -1
        R_optimal = np.dot(Vt.T, U.T)

    # Step 5: Compute translation
    translation = centroid_B - np.dot(R_optimal, centroid_A)

    return R_optimal, translation