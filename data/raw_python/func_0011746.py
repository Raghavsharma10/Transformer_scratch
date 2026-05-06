def _scalar_to_vector(self, m):
        """Allow submodels with scalar equations. Convert to 1D vector systems.
        Args:
          m (Model)
        """
        if not isinstance(m.y0, numbers.Number):
            return m
        else:
            m = copy.deepcopy(m)
            t0 = 0.0
            if isinstance(m.y0, numbers.Integral):
                numtype = np.float64
            else:
                numtype = type(m.y0)
            y0_orig = m.y0
            m.y0 = np.array([m.y0], dtype=numtype)
            def make_vector_fn(fn):
                def newfn(y, t):
                    return np.array([fn(y[0], t)], dtype=numtype)
                newfn.__name__ = fn.__name__
                return newfn
            def make_matrix_fn(fn):
                def newfn(y, t):
                    return np.array([[fn(y[0], t)]], dtype=numtype)
                newfn.__name__ = fn.__name__
                return newfn
            def make_coupling_fn(fn):
                def newfn(source_y, target_y, weight):
                    return np.array([fn(source_y[0], target_y[0], weight)])
                newfn.__name__ = fn.__name__
                return newfn
            if isinstance(m.f(y0_orig, t0), numbers.Number):
                m.f = make_vector_fn(m.f)
            if hasattr(m, 'G') and isinstance(m.G(y0_orig,t0), numbers.Number):
                m.G = make_matrix_fn(m.G)
            if (hasattr(m, 'coupling') and
                    isinstance(m.coupling(y0_orig, y0_orig, 0.5),
                               numbers.Number)):
                m.coupling = make_coupling_fn(m.coupling)
            return m