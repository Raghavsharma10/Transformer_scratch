def mountain_car_trajectories(num_traj):
  '''Collect data using random hard-coded policies on MountainCar.

  num_traj : int, number of trajectories to collect

  Returns (trajectories, traces)
  '''
  domain = MountainCar()
  slopes = np.random.normal(0, 0.01, size=num_traj)
  v0s = np.random.normal(0, 0.005, size=num_traj)
  trajectories = []
  traces = []
  norm = np.array((domain.MAX_POS-domain.MIN_POS,
                   domain.MAX_VEL-domain.MIN_VEL))
  for m,b in zip(slopes, v0s):
    mcar_policy = lambda s: 0 if s[0]*m + s[1] + b > 0 else 2
    start = (np.random.uniform(domain.MIN_POS,domain.MAX_POS),
             np.random.uniform(domain.MIN_VEL,domain.MAX_VEL))
    samples = _run_episode(mcar_policy, domain, start, max_iters=40)
    # normalize
    samples.state /= norm
    samples.next_state /= norm
    traces.append(samples)
    if samples.reward[-1] == 0:
      # Don't include the warp to the final state.
      trajectories.append(samples.state[:-1])
    else:
      trajectories.append(samples.state)

  return trajectories, traces