def cv_residuals(self, cv=True):
        """Return the residuals of the cross-validation for the fit data"""
        vals = self.cv_values(cv)
        return (self.y - vals) / self.dy