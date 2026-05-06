def cv_error(self, cv=True, skip_endpoints=True):
        """Return the sum of cross-validation residuals for the input data"""
        resids = self.cv_residuals(cv)
        if skip_endpoints:
            resids = resids[1:-1]
        return np.mean(abs(resids))