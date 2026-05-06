def orient_to(self, other, index=None, inplace=False, scaling=False):
        """Orient this Projection to another dataset.

        Orient this projection using reflection, rotation and translation to
        match another projection using procrustes superimposition. Scaling is
        optional.

        Args:
            other
                (:py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                or `array-like`):
                The other dataset to orient this projection to.
                If other is an instance of :py:class:`pymds.Projection` or
                :py:class:`pandas.DataFrame`, then other must have indexes in
                common with this projection. If `array-like`, then other must
                have the same dimensions as self.coords.
            index (`list-like` or `None`): If other is an instance of
                :py:class:`pandas.DataFrame` or :py:class:`pymds.Projection`
                then orient this projection to other using only samples in
                index.
            inplace (`bool`): Update coordinates of this projection inplace,
                or return an instance of :py:class:`pymds.Projection`.
            scaling (`bool`): Allow scaling. (Not implemented yet).

        Examples:

            .. doctest::

                >>> import numpy as np
                >>> import pandas as pd
                >>> from pymds import Projection
                >>> array = np.random.randn(10, 2)
                >>> pro = Projection(pd.DataFrame(array))
                >>> # Flip left-right, rotate 90 deg and translate
                >>> other = np.fliplr(array)
                >>> other = np.dot(other, np.array([[0, -1], [1, 0]]))
                >>> other += np.array([10, -5])
                >>> oriented = pro.orient_to(other)
                >>> (oriented.coords.values - other).sum() < 1e-6
                True

        Returns:
            :py:class:`pymds.Projection`: If ``inplace=False``.
        """
        arr_self, arr_other = self._get_samples_shared_with(other, index=index)

        if scaling:
            raise NotImplementedError()

        else:
            self_mean = arr_self.mean(axis=0)
            other_mean = arr_other.mean(axis=0)

            A = arr_self - self_mean
            B = arr_other - other_mean
            R, _ = orthogonal_procrustes(A, B)

            to_rotate = self.coords - self.coords.mean(axis=0)
            oriented = np.dot(to_rotate, R) + other_mean
            oriented = pd.DataFrame(oriented, index=self.coords.index)

        if inplace:
            self.coords = oriented
        else:
            return Projection(oriented)