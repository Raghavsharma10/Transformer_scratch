def _principal_angle(a, B):
  '''a is (d,t), B is (k,d,t)'''
  # TODO: check case for t = d-1
  if a.shape[1] == 1:
    return a.T.dot(B)[0,:,0]

  # find normals that maximize distance when projected
  x1 = np.einsum('abc,adc->abd', B, B).dot(a) - a   # b.dot(b.T).dot(a) - a
  x2 = np.einsum('ab,cad->cbd', a.dot(a.T), B) - B  # a.dot(a.T).dot(b) - b
  xx = np.vstack((x1, x2))

  # batch PCA (1st comp. only)
  xx -= xx.mean(axis=1)[:,None]
  c = np.einsum('abc,abd->acd', xx, xx)
  _, vecs = np.linalg.eigh(c)
  fpc = vecs[:,:,-1]
  fpc1 = fpc[:len(x1)]
  fpc2 = fpc[len(x1):]

  # a.dot(fpc1).dot(b.dot(fpc2))
  lhs = a.dot(fpc1.T).T
  rhs = np.einsum('abc,ac->ab', B, fpc2)
  return np.einsum('ij,ij->i', lhs, rhs)