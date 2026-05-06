def _transform_row(self, in_row, out_row):
        """
        Transforms an input row to an output row (i.e. (partial) dimensional data).

        :param dict[str,str] in_row: The input row.
        :param dict[str,T] out_row: The output row.

        :rtype: (str,str)
        """
        tmp_row = {}

        for step in self._steps:
            park_info, ignore_info = step(in_row, tmp_row, out_row)
            if park_info or ignore_info:
                return park_info, ignore_info

        return None, None