def barycenter_edge_weights(self, X, copy=True, reg=1e-3):
    '''Re-weight such that the sum of each vertex's edge weights is 1.
    The resulting weighted graph is suitable for locally linear embedding.
    reg : amount of regularization to keep the problem well-posed
    '''
    new_weights = []
    for i, adj in enumerate(self.adj_list()):
      C = X[adj] - X[i]
      G = C.dot(C.T)
      trace = np.trace(G)
      r = reg * trace if trace > 0 else reg
      G.flat[::G.shape[1] + 1] += r
      w = solve(G, np.ones(G.shape[0]), sym_pos=True,
                overwrite_a=True, overwrite_b=True)
      w /= w.sum()
      new_weights.extend(w.tolist())
    return self.reweight(new_weights, copy=copy)