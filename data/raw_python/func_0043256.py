def plot_lines_to(self, other, index=None, **kwds):
        """Plot lines from samples shared between this projection and another
        dataset.

        Args:
            other
                (:py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                or `array-like`):
                The other dataset to plot lines to. If other is an instance of
                :py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`,
                then other must have indexes in common with this projection.
                If `array-like`, then other must have the same dimensions as
                `self.coords`.
            index (`list-like` or `None`): Only draw lines between samples in
                index. All elements in index must be samples in this projection
                and other.
            **kwds: Passed to :py:obj:`matplotlib.collections.LineCollection`.

        Examples:

            >>> import numpy as np
            >>> from pymds import Projection
            >>> pro = Projection(np.random.randn(50, 2))
            >>> R = np.array([[0, -1], [1, 0]])
            >>> other = np.dot(pro.coords, R)  # Rotate 90 deg
            >>> ax = pro.plot(c='black', edgecolor='white', zorder=20)
            >>> ax = pro.plot_lines_to(other, linewidths=0.3)

        Returns:
            :py:obj:`matplotlib.axes.Axes`
        """
        start, end = self._get_samples_shared_with(other, index=index)
        segments = [[start[i, :], end[i, :]] for i in range(start.shape[0])]
        ax = plt.gca()
        ax.add_artist(LineCollection(segments=segments, **kwds))
        return ax