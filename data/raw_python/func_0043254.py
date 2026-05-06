def _get_samples_shared_with(self, other, index=None):
        """Find samples shared with another dataset.

        Args:
            other
                (:py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                    or `array-like`):
                The other dataset. If `other` is an instance of
                :py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`,
                then `other` must have indexes in common with this projection.
                If `array-like`, then other must have same dimensions as
                `self.coords`.
            index (`list-like` or `None`): If `other` is an instance of
                :py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                then only return samples in index.

        Returns:
            `tuple`: containing:

                - this (`numpy.array`) Shape [`x`, `n`].
                - other (`numpy.array`) Shape [`x`, `n`].
        """
        if isinstance(other, (pd.DataFrame, Projection)):
            df_other = other.coords if isinstance(other, Projection) else other

            if len(set(df_other.index)) != len(df_other.index):
                raise ValueError("other index has duplicates")

            if len(set(self.coords.index)) != len(self.coords.index):
                raise ValueError("This projection index has duplicates")

            if index:
                uniq_idx = set(index)

                if len(uniq_idx) != len(index):
                    raise ValueError("index has has duplicates")

                if uniq_idx - set(df_other.index):
                    raise ValueError("index has samples not in other")

                if uniq_idx - set(self.coords.index):
                    raise ValueError(
                        "index has samples not in this projection")

            else:
                uniq_idx = set(df_other.index) & set(self.coords.index)

                if not len(uniq_idx):
                    raise ValueError(
                        "No samples shared between other and this projection")

            idx = list(uniq_idx)
            return self.coords.loc[idx, :].values, df_other.loc[idx, :].values

        else:
            other = np.array(other)

            if other.shape != self.coords.shape:
                raise ValueError(
                    "array-like must have the same shape as self.coords")

            return self.coords.values, other