def baart_criteria(self, X, y):
        """
        Returns the optimal Fourier series degree as determined by
        `Baart's Criteria <http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1986A%26A...170...59P&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf>`_ [JOP]_.

        **Citations**

        .. [JOP] J. O. Petersen, 1986,
                 "Studies of Cepheid type variability. IV.
                 The uncertainties of Fourier decomposition parameters.",
                 A&A, Vol. 170, p. 59-69
        """
        try:
            min_degree, max_degree = self.degree_range
        except ValueError:
            raise ValueError("Degree range must be a length two sequence")

        cutoff = self.baart_tolerance(X)
        pipeline = Pipeline([('Fourier', Fourier()),
                             ('Regressor', self.regressor)])
        sorted_X = numpy.sort(X, axis=0)
        X_sorting = numpy.argsort(rowvec(X))
        for degree in range(min_degree, max_degree):
            pipeline.set_params(Fourier__degree=degree)
            pipeline.fit(X, y)
            lc = pipeline.predict(sorted_X)
            residuals = y[X_sorting] - lc
            p_c = autocorrelation(residuals)
            if abs(p_c) <= cutoff:
                return degree
        # reached max_degree without reaching cutoff
        return max_degree