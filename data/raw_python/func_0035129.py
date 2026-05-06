def epsilon_net(points, close_distance):
  '''Selects a subset of `points` to preserve graph structure while minimizing
  the number of points used, by removing points within `close_distance`.
  Returns the downsampled indices.'''
  num_points = points.shape[0]
  indices = set(range(num_points))
  selected = []
  while indices:
    idx = indices.pop()
    nn_inds, = nearest_neighbors(points[idx], points, epsilon=close_distance)
    indices.difference_update(nn_inds)
    selected.append(idx)
  return selected