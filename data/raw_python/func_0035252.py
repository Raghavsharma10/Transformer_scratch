def edge_cluster_angle(edge_dirs, subspaces1, subspaces2):
  '''edge_dirs is a (n,D) matrix of edge vectors.
  subspaces are (n,D,d) or (D,d) matrices of normalized orthogonal subspaces.
  Result is an n-length array of angles.'''
  QG = edge_dirs / np.linalg.norm(edge_dirs, ord=2, axis=1)[:,None]
  X1 = np.einsum('...ij,...i->...j', subspaces1, QG)
  X2 = np.einsum('...ij,...i->...j', subspaces2, QG)
  # TODO: check the math on this for more cases
  # angles = np.maximum(1-np.sum(X1**2, axis=1), 1-np.sum(X2**2, axis=1))
  C1 = np.linalg.svd(X1[:,:,None], compute_uv=False)
  C2 = np.linalg.svd(X2[:,:,None], compute_uv=False)
  angles = np.maximum(1-C1**2, 1-C2**2)[:,0]
  angles[np.isnan(angles)] = 0.0  # nan when edge length == 0
  return angles