def bounding_box(self, factor=10.0):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        .. math::

            x_{\\textnormal{low}} = 0

            x_{\\textnormal{high}} = \\log(\\lambda_{\\textnormal{max}} \\;\
            (1 + \\textnormal{factor}))

        Parameters
        ----------
        factor : float
            Used to calculate ``x_high``.

        """
        w0 = self.lambda_max
        return (w0 * 0, np.log10(w0 + factor * w0))