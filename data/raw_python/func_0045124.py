def mtakfac(m):
  """Modified Takagi factorization of a (complex) symmetric matrix.

  Returns U, S where U^T M U = diag(S) and the singular values
  are sorted in ascending order (small to large).
  """
  u, s, v = msvd(np.asarray(m, dtype=complex))
  f = np.sqrt(u.conj().T @ v.conj())
  w = v @ f
  return w, s