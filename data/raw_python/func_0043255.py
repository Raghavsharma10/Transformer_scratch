def plot(self, **kwds):
        """Plot the coordinates in the first two dimensions of the projection.

        Removes axis and tick labels, and sets the grid spacing to 1 unit.
        One way to display the grid is to use `Seaborn`_:

        Args:
            **kwds: Passed to :py:meth:`pandas.DataFrame.plot.scatter`.

        Examples:

            >>> from pymds import DistanceMatrix
            >>> import pandas as pd
            >>> import seaborn as sns
            >>> sns.set_style('whitegrid')
            >>> dist = pd.DataFrame({
            ...    'a': [0.0, 1.0, 2.0],
            ...    'b': [1.0, 0.0, 3 ** 0.5],
            ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
            >>> dm = DistanceMatrix(dist)
            >>> pro = dm.optimize()
            >>> ax = pro.plot(c='black', s=50, edgecolor='white')

        Returns:
            :py:obj:`matplotlib.axes.Axes`

        .. _Seaborn:
            https://seaborn.pydata.org/
        """
        ax = plt.gca()
        self.coords.plot.scatter(x=0, y=1, ax=ax, **kwds)
        ax.get_xaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.get_yaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect(1)
        return ax