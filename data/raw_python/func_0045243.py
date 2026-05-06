def _merge_adjacent_rows(self, rows):
        """
        Resolves adjacent and overlapping rows. Overlapping rows are resolved as follows:
        * The interval with the most recent begin date prevails for the overlapping period.
        * If the begin dates are the same the interval with the most recent end date prevails.
        * If the begin and end dates are equal the last row in the data set prevails.
        Identical (excluding begin and end date) adjacent rows are replace with a single row.

        :param list[dict[str,T]] rows: The rows in a group (i.e. with the same natural key).
        .
        :rtype: list[dict[str,T]]
        """
        ret = list()

        prev_row = None
        for row in rows:
            if prev_row:
                relation = Allen.relation(prev_row[self._key_start_date],
                                          prev_row[self._key_end_date],
                                          row[self._key_start_date],
                                          row[self._key_end_date])
                if relation is None:
                    # row holds an invalid interval (prev_row always holds a valid interval). Hence, the join is empty.
                    return []

                elif relation == Allen.X_BEFORE_Y:
                    # Two rows with distinct intervals.
                    # prev_row: |----|
                    # row:                 |-----|
                    ret.append(prev_row)
                    prev_row = row

                elif relation == Allen.X_MEETS_Y:
                    # The two rows are adjacent.
                    # prev_row: |-------|
                    # row:               |-------|
                    if self._equal(prev_row, row):
                        # The two rows are identical (except for start and end date) and adjacent. Combine the two rows
                        # into one row.
                        prev_row[self._key_end_date] = row[self._key_end_date]
                    else:
                        # Rows are adjacent but not identical.
                        ret.append(prev_row)
                        prev_row = row

                elif relation == Allen.X_OVERLAPS_WITH_Y:
                    # prev_row overlaps row. Should not occur with proper reference data.
                    # prev_row: |-----------|
                    # row:            |----------|
                    if self._equal(prev_row, row):
                        # The two rows are identical (except for start and end date) and overlapping. Combine the two
                        # rows into one row.
                        prev_row[self._key_end_date] = row[self._key_end_date]
                    else:
                        # Rows are overlapping but not identical.
                        prev_row[self._key_end_date] = row[self._key_start_date] - 1
                        ret.append(prev_row)
                        prev_row = row

                elif relation == Allen.X_STARTS_Y:
                    # prev_row start row. Should not occur with proper reference data.
                    # prev_row: |------|
                    # row:      |----------------|
                    prev_row = row

                elif relation == Allen.X_EQUAL_Y:
                    # Can happen when the reference data sets are joined without respect for date intervals.
                    # prev_row: |----------------|
                    # row:      |----------------|
                    prev_row = row

                elif relation == Allen.X_DURING_Y_INVERSE:
                    # row during prev_row. Should not occur with proper reference data.
                    # prev_row: |----------------|
                    # row:           |------|
                    # Note: the interval with the most recent start date prevails. Hence, the interval after
                    # row[self._key_end_date] is discarded.
                    if self._equal(prev_row, row):
                        prev_row[self._key_end_date] = row[self._key_end_date]
                    else:
                        prev_row[self._key_end_date] = row[self._key_start_date] - 1
                        ret.append(prev_row)
                        prev_row = row

                elif relation == Allen.X_FINISHES_Y_INVERSE:
                    # row finishes prev_row. Should not occur with proper reference data.
                    # prev_row: |----------------|
                    # row:                |------|
                    if not self._equal(prev_row, row):
                        prev_row[self._key_end_date] = row[self._key_start_date] - 1
                        ret.append(prev_row)
                        prev_row = row

                        # Note: if the two rows are identical (except for start and end date) nothing to do.
                else:
                    # Note: The rows are sorted such that prev_row[self._key_begin_date] <= row[self._key_begin_date].
                    # Hence the following relation should not occur: X_DURING_Y,  X_FINISHES_Y, X_BEFORE_Y_INVERSE,
                    # X_MEETS_Y_INVERSE, X_OVERLAPS_WITH_Y_INVERSE, and X_STARTS_Y_INVERSE. Hence, we covered all 13
                    # relations in Allen's interval algebra.
                    raise ValueError('Data is not sorted properly. Relation: {0}'.format(relation))

            elif row[self._key_start_date] <= row[self._key_end_date]:
                # row is the first valid row.
                prev_row = row

        if prev_row:
            ret.append(prev_row)

        return ret