def condense_duplicates(self):
        """Condense duplicate points using a transformation matrix.
        
        This is useful if you have multiple non-transformed points at the same
        location or multiple transformed points that use the same quadrature
        points.
        
        Won't change the GP if all of the rows of [X, n] are unique. Will create
        a transformation matrix T if necessary. Note that the order of the
        points in [X, n] will be arbitrary after this operation.
        
        If there are any transformed quantities (i.e., `self.T` is not None), it
        will also remove any quadrature points for which all of the weights are
        zero (even if all of the rows of [X, n] are unique).
        """
        unique, inv = unique_rows(
            scipy.hstack((self.X, self.n)),
            return_inverse=True
        )
        # Only proceed if there is anything to be gained:
        if len(unique) != len(self.X):
            if self.T is None:
                self.T = scipy.eye(len(self.y))
            new_T = scipy.zeros((len(self.y), unique.shape[0]))
            for j in xrange(0, len(inv)):
                new_T[:, inv[j]] += self.T[:, j]
            self.T = new_T
            self.n = unique[:, self.X.shape[1]:]
            self.X = unique[:, :self.X.shape[1]]
        # Also remove any points which don't enter into the calculation:
        if self.T is not None:
            # Find the columns of T which actually enter in:
            # Recall that T is (n, n_Q), X is (n_Q, n_dim).
            good_cols = (self.T != 0.0).any(axis=0)
            self.T = self.T[:, good_cols]
            self.X = self.X[good_cols, :]
            self.n = self.n[good_cols, :]