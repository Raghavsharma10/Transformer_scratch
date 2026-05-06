def add_data(self, X, y, err_y=0, n=0, T=None):   
        """Add data to the training data set of the GaussianProcess instance.
        
        Parameters
        ----------
        X : array, (`M`, `D`)
            `M` input values of dimension `D`.
        y : array, (`M`,)
            `M` target values.
        err_y : array, (`M`,) or scalar float, optional
            Non-negative values only. Error given as standard deviation) in the
            `M` target values. If `err_y` is a scalar, the data set is taken to
            be homoscedastic (constant error). Otherwise, the length of `err_y`
            must equal the length of `y`. Default value is 0 (noiseless
            observations).
        n : array, (`M`, `D`) or scalar float, optional
            Non-negative integer values only. Degree of derivative for each
            target. If `n` is a scalar it is taken to be the value for all
            points in `y`. Otherwise, the length of n must equal the length of
            `y`. Default value is 0 (observation of target value). If
            non-integer values are passed, they will be silently rounded.
        T : array, (`M`, `N`), optional
            Linear transformation to get from latent variables to data in the
            argument `y`. When `T` is passed the argument `y` holds the
            transformed quantities `y=TY(X)` where `y` are the observed values
            of the transformed quantities, `T` is the transformation matrix and
            `Y(X)` is the underlying (untransformed) values of the function to
            be fit that enter into the transformation. When `T` is `M`-by-`N`
            and `y` has `M` elements, `X` and `n` will both be `N`-by-`D`.
            Default is None (no transformation).
        
        Raises
        ------
        ValueError
            Bad shapes for any of the inputs, negative values for `err_y` or `n`.
        """
        # Verify y has only one non-trivial dimension:
        y = scipy.atleast_1d(scipy.asarray(y, dtype=float))
        if len(y.shape) != 1:
            raise ValueError(
                "Training targets y must have only one dimension with length "
                "greater than one! Shape of y given is %s" % (y.shape,)
            )
        
        # Handle scalar error or verify shape of array error matches shape of y:
        try:
            iter(err_y)
        except TypeError:
            err_y = err_y * scipy.ones_like(y, dtype=float)
        else:
            err_y = scipy.asarray(err_y, dtype=float)
            if err_y.shape != y.shape:
                raise ValueError(
                    "When using array-like err_y, shape must match shape of y! "
                    "Shape of err_y given is %s, shape of y given is %s." % (err_y.shape, y.shape)
                )
        if (err_y < 0).any():
            raise ValueError("All elements of err_y must be non-negative!")
        
        # Handle scalar training input or convert array input into 2d.
        X = scipy.atleast_2d(scipy.asarray(X, dtype=float))
        # Correct single-dimension inputs:
        if self.num_dim == 1 and X.shape[0] == 1:
            X = X.T
        if T is None and X.shape != (len(y), self.num_dim):
            raise ValueError(
                "Shape of training inputs must be (len(y), k.num_dim)! X given "
                "has shape %s, shape of y is %s and num_dim=%d." % (X.shape, y.shape, self.num_dim)
            )
        
        # Handle scalar derivative orders or verify shape of array derivative
        # orders matches shape of y:
        try:
            iter(n)
        except TypeError:
            n = n * scipy.ones_like(X, dtype=int)
        else:
            n = scipy.atleast_2d(scipy.asarray(n, dtype=int))
            # Correct single-dimension inputs:
            if self.num_dim == 1 and n.shape[1] != 1:
                n = n.T
            if n.shape != X.shape:
                raise ValueError(
                    "When using array-like n, shape must be (len(y), k.num_dim)! "
                    "Shape of n given is %s, shape of y given is %s and num_dim=%d."
                    % (n.shape, y.shape, self.num_dim)
                )
        if (n < 0).any():
            raise ValueError("All elements of n must be non-negative integers!")
        
        # Handle transform:
        if T is None and self.T is not None:
            T = scipy.eye(len(y))
        if T is not None:
            T = scipy.atleast_2d(scipy.asarray(T, dtype=float))
            if T.ndim != 2:
                raise ValueError("T must have exactly 2 dimensions!")
            if T.shape[0] != len(y):
                raise ValueError(
                    "T must have as many rows are there are elements in y!"
                )
            if T.shape[1] != X.shape[0]:
                raise ValueError(
                    "There must be as many columns in T as there are rows in X!"
                )
            if self.T is None and self.X is not None:
                self.T = scipy.eye(len(self.y))
            
            if self.T is None:
                self.T = T
            else:
                self.T = scipy.linalg.block_diag(self.T, T)
        
        if self.X is None:
            self.X = X
        else:
            self.X = scipy.vstack((self.X, X))
        self.y = scipy.append(self.y, y)
        self.err_y = scipy.append(self.err_y, err_y)
        if self.n is None:
            self.n = n
        else:
            self.n = scipy.vstack((self.n, n))
        self.K_up_to_date = False