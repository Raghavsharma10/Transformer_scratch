def layout_spring(self, num_dims=2, spring_constant=None, iterations=50,
                    initial_temp=0.1, initial_layout=None):
    '''Position vertices using the Fruchterman-Reingold (spring) algorithm.

    num_dims : int (default=2)
       Number of dimensions to embed vertices in.

    spring_constant : float (default=None)
       Optimal distance between nodes.  If None the distance is set to
       1/sqrt(n) where n is the number of nodes.  Increase this value
       to move nodes farther apart.

    iterations : int (default=50)
       Number of iterations of spring-force relaxation

    initial_temp : float (default=0.1)
       Largest step-size allowed in the dynamics, decays linearly.
       Must be positive, should probably be less than 1.

    initial_layout : array-like of shape (n, num_dims)
       If provided, serves as the initial placement of vertex coordinates.
    '''
    if initial_layout is None:
      X = np.random.random((self.num_vertices(), num_dims))
    else:
      X = np.array(initial_layout, dtype=float, copy=True)
      assert X.shape == (self.num_vertices(), num_dims)
    if spring_constant is None:
      # default to sqrt(area_of_viewport / num_vertices)
      spring_constant = X.shape[0] ** -0.5
    S = self.matrix('csr', 'csc', 'coo', copy=True)
    S.data[:] = 1. / S.data  # Convert to similarity
    ii,jj = S.nonzero()  # cache nonzero indices
    # simple cooling scheme, linearly steps down
    cooling_scheme = np.linspace(initial_temp, 0, iterations+2)[:-2]
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for t in cooling_scheme:
      delta = X[:,None] - X[None]
      distance = _bounded_norm(delta, 1e-8)
      # repulsion from all vertices
      force = spring_constant**2 / distance
      # attraction from connected vertices
      force[ii,jj] -= S.data * distance[ii,jj]**2 / spring_constant
      displacement = np.einsum('ijk,ij->ik', delta, force)
      # update positions
      length = _bounded_norm(displacement, 1e-2)
      X += displacement * t / length[:,None]
    return X