def __ordered_values_by_indexes(self, data, inds):
        """
        Return values (intensities) by indexes.

        Used for multiscale graph cut.
        data = [[0 1 1],
                [0 2 2],
                [0 2 2]]

        inds = [[0 1 2],
                [3 4 4],
                [5 4 4]]

        return: [0, 1, 1, 0, 2, 0]

        If the data are not consistent, it will take the maximal value

        """
        # get unique labels and their first indexes
        # lab, linds = np.unique(inds, return_index=True)
        # compute values by indexes
        # values = data.reshape(-1)[linds]

        # alternative slow implementation
        # if there are different data on same index, it will take
        # maximal value
        # lab = np.unique(inds)
        # values = [0]*len(lab)
        # for label in lab:
        #     values[label] = np.max(data[inds == label])
        #
        # values = np.asarray(values)

        # yet another implementation
        values = [None] * (np.max(inds) + 1)

        linear_inds = inds.ravel()
        linear_data = data.ravel()
        for i in range(0, len(linear_inds)):
            # going over all data pixels

            if values[linear_inds[i]] is None:
                # this index is found for first
                values[linear_inds[i]] = linear_data[i]
            elif values[linear_inds[i]] < linear_data[i]:
                # here can be changed maximal or minimal value
                values[linear_inds[i]] = linear_data[i]

        values = np.asarray(values)

        return values