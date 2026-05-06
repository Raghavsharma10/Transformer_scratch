def _updateB(oldB, B, W, degrees, damping, inds, backinds):  # pragma: no cover
  '''belief update function.'''
  for j,d in enumerate(degrees):
    kk = inds[j]
    bk = backinds[j]

    if d == 0:
      B[kk,bk] = -np.inf
      continue

    belief = W[kk,bk] + W[j]
    oldBj = oldB[j]
    if d == oldBj.shape[0]:
      bth = quickselect(-oldBj, d-1)
      bplus = -1
    else:
      bth,bplus = quickselect(-oldBj, d-1, d)

    belief -= np.where(oldBj >= oldBj[bth], oldBj[bplus], oldBj[bth])
    B[kk,bk] = damping*belief + (1-damping)*oldB[kk,bk]