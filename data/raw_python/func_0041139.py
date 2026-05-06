def integrate_box(self,low,high,forcequad=False,**kwargs):
        """Integrates over a box. Optionally force quad integration, even for non-adaptive.

        If adaptive mode is not being used, this will just call the
        `scipy.stats.gaussian_kde` method `integrate_box_1d`.  Else,
        by default, it will call `scipy.integrate.quad`.  If the
        `forcequad` flag is turned on, then that integration will be
        used even if adaptive mode is off.

        Parameters
        ----------
        low : float
            Lower limit of integration

        high : float
            Upper limit of integration

        forcequad : bool
            If `True`, then use the quad integration even if adaptive mode is off.

        kwargs
            Keyword arguments passed to `scipy.integrate.quad`.
        """
        if not self.adaptive and not forcequad:
            return self.gauss_kde.integrate_box_1d(low,high)*self.norm
        return quad(self.evaluate,low,high,**kwargs)[0]