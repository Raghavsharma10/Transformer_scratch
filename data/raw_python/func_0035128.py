def downsample_trajectories(trajectories, downsampler, *args, **kwargs):
  '''Downsamples all points together, then re-splits into original trajectories.

  trajectories : list of 2-d arrays, each representing a trajectory
  downsampler(X, *args, **kwargs) : callable that returns indices into X
  '''
  X = np.vstack(trajectories)
  traj_lengths = list(map(len, trajectories))
  inds = np.sort(downsampler(X, *args, **kwargs))
  new_traj = []
  for stop in np.cumsum(traj_lengths):
    n = np.searchsorted(inds, stop)
    new_traj.append(X[inds[:n]])
    inds = inds[n:]
  return new_traj