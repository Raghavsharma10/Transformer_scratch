def b_matching(D, k, max_iter=1000, damping=1, conv_thresh=1e-4,
               weighted=False, verbose=False):
  '''
  "Belief-Propagation for Weighted b-Matchings on Arbitrary Graphs
  and its Relation to Linear Programs with Integer Solutions"
  Bayati et al.

  Finds the minimal weight perfect b-matching using min-sum loopy-BP.

  @param D pairwise distance matrix
  @param k number of neighbors per vertex (scalar or array-like)

  Based on the code at http://www.cs.columbia.edu/~bert/code/bmatching/bdmatch
  '''
  INTERVAL = 2
  oscillation = 10
  cbuff = np.zeros(100, dtype=float)
  cbuffpos = 0
  N = D.shape[0]
  assert D.shape[1] == N, 'Input distance matrix must be square'
  mask = ~np.eye(N, dtype=bool)  # Assume all nonzero except for diagonal
  W = -D[mask].reshape((N, -1)).astype(float)
  degrees = np.clip(np.atleast_1d(k), 0, N-1)
  if degrees.size == 1:  # broadcast scalar up to length-N array
    degrees = np.repeat(degrees, N)
  else:
    assert degrees.shape == (N,), 'Input degrees must have length N'
  # TODO: remove these later
  inds = np.tile(np.arange(N), (N, 1))
  backinds = inds.copy()
  inds = inds[mask].reshape((N, -1))
  backinds = backinds.T.ravel()[:(N*(N-1))].reshape((N, -1))

  # Run Belief Revision
  change = 1.0
  B = W.copy()
  for n_iter in range(1, max_iter+1):
    oldB = B.copy()
    update_belief(oldB, B, W, degrees, damping, inds, backinds)

    # check for convergence
    if n_iter % INTERVAL == 0:
      # track changes
      c = np.abs(B[:,0]).sum()
      # c may be infinite here, and that's ok
      with np.errstate(invalid='ignore'):
        if np.any(np.abs(c - cbuff) < conv_thresh):
          oscillation -= 1
      cbuff[cbuffpos] = c
      cbuffpos = (cbuffpos + 1) % len(cbuff)

      change = diff_belief(B, oldB)
      if np.isnan(change):
        warnings.warn("change is NaN! "
                      "BP will quit but solution could be invalid. "
                      "Problem may be infeasible.")
        break
      if change < conv_thresh or oscillation < 1:
        break
  else:
    warnings.warn("Hit iteration limit (%d) before converging" % max_iter)

  if verbose:  # pragma: no cover
    if change < conv_thresh:
      print("Converged to stable beliefs in %d iterations" % n_iter)
    elif oscillation < 1:
      print("Stopped after reaching oscillation in %d iterations" % n_iter)
      print("No feasible solution found or there are multiple maxima.")
      print("Outputting best approximate solution. Try damping.")

  # recover result from B
  thresholds = np.zeros(N)
  for i,d in enumerate(degrees):
    Brow = B[i]
    if d >= N - 1:
      thresholds[i] = -np.inf
    elif d < 1:
      thresholds[i] = np.inf
    else:
      thresholds[i] = Brow[quickselect(-Brow, d-1)]

  ii,jj = np.where(B >= thresholds[:,None])
  pairs = np.column_stack((ii, inds[ii,jj]))
  w = D[ii, pairs[:,1]] if weighted else None
  return Graph.from_edge_pairs(pairs, num_vertices=N, weights=w)