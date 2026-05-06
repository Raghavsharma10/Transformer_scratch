def _calculate_delta_pos(adjacency_arr, pos, t, optimal):
    """Helper to calculate the delta position"""
    # XXX eventually this should be refactored for the sparse case to only
    # do the necessary pairwise distances
    delta = pos[:, np.newaxis, :] - pos

    # Distance between points
    distance2 = (delta*delta).sum(axis=-1)
    # Enforce minimum distance of 0.01
    distance2 = np.where(distance2 < 0.0001, 0.0001, distance2)
    distance = np.sqrt(distance2)
    # Displacement "force"
    displacement = np.zeros((len(delta), 2))
    for ii in range(2):
        displacement[:, ii] = (
            delta[:, :, ii] *
            ((optimal * optimal) / (distance*distance) -
             (adjacency_arr * distance) / optimal)).sum(axis=1)

    length = np.sqrt((displacement**2).sum(axis=1))
    length = np.where(length < 0.01, 0.1, length)
    delta_pos = displacement * t / length[:, np.newaxis]
    return delta_pos