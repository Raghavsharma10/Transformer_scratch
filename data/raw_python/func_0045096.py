def _distinct(row1, row2):
        """
        Returns a list of distinct (or none overlapping) intervals if two intervals are overlapping. Returns None if
        the two intervals are none overlapping. The list can have 2 or 3 intervals.

        :param tuple[int,int] row1: The first interval.
        :param tuple[int,int] row2: The second interval.

        :rtype: None|list(tuple[int,int])
        """
        relation = Allen.relation(row1[0], row1[1], row2[0], row2[1])

        if relation is None:
            # One of the 2 intervals is invalid.
            return []

        if relation == Allen.X_BEFORE_Y:
            # row1: |----|
            # row2:            |-----|
            return None  # [(row1[0], row1[1]), (row2[0], row2[1])]

        if relation == Allen.X_BEFORE_Y_INVERSE:
            # row1:            |-----|
            # row2: |----|
            return None  # [(row2[0], row2[1]), (row1[0], row1[1])]

        if relation == Allen.X_MEETS_Y:
            # row1: |-------|
            # row2:          |-------|
            return None  # [(row1[0], row1[1]), (row2[0], row2[1])]

        if relation == Allen.X_MEETS_Y_INVERSE:
            # row1:          |-------|
            # row2: |-------|
            return None  # [(row2[0], row2[1]), (row1[0], row1[1])]

        if relation == Allen.X_OVERLAPS_WITH_Y:
            # row1: |-----------|
            # row2:       |----------|
            return [(row1[0], row2[0] - 1), (row2[0], row1[1]), (row1[1] + 1, row2[1])]

        if relation == Allen.X_OVERLAPS_WITH_Y_INVERSE:
            # row1:       |----------|
            # row2: |-----------|
            return [(row2[0], row1[0] - 1), (row1[0], row2[1]), (row2[1] + 1, row1[1])]

        if relation == Allen.X_STARTS_Y:
            # row1: |------|
            # row2: |----------------|
            return [(row1[0], row1[1]), (row1[1] + 1, row2[1])]

        if relation == Allen.X_STARTS_Y_INVERSE:
            # row1: |----------------|
            # row2: |------|
            return [(row2[0], row2[1]), (row2[1] + 1, row1[1])]

        if relation == Allen.X_DURING_Y:
            # row1:      |------|
            # row2: |----------------|
            return [(row2[0], row1[0] - 1), (row1[0], row1[1]), (row1[1] + 1, row2[1])]

        if relation == Allen.X_DURING_Y_INVERSE:
            # row1: |----------------|
            # row2:      |------|
            return [(row1[0], row2[0] - 1), (row2[0], row2[1]), (row2[1] + 1, row1[1])]

        if relation == Allen.X_FINISHES_Y:
            # row1:           |------|
            # row2: |----------------|
            return [(row2[0], row1[0] - 1), (row1[0], row1[1])]

        if relation == Allen.X_FINISHES_Y_INVERSE:
            # row1: |----------------|
            # row2:           |------|
            return [(row1[0], row2[0] - 1), (row2[0], row2[1])]

        if relation == Allen.X_EQUAL_Y:
            # row1: |----------------|
            # row2: |----------------|
            return None  # [(row1[0], row1[1])]

        # We got all 13 relation in Allen's interval algebra covered.
        raise ValueError('Unexpected relation {0}'.format(relation))