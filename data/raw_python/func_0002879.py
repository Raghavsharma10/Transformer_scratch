def _in_header(self, col):
        """Validate column identifier(s)."""
        # pylint: disable=R1704
        if not self._has_header:
            # Conditionally register exceptions so that they do not appear
            # in situations where has_header is always True. In this way
            # they are not auto-documented by default
            icol_ex = pexdoc.exh.addex(RuntimeError, "Invalid column specification")
        hnf_ex = pexdoc.exh.addex(ValueError, "Column *[column_identifier]* not found")
        col_list = [col] if isinstance(col, (str, int)) else col
        for col in col_list:
            edata = {"field": "column_identifier", "value": col}
            if not self._has_header:
                # Condition not subsumed in raise_exception_if
                # so that calls that always have has_header=True
                # do not pick up this exception
                icol_ex(not isinstance(col, int))
                hnf_ex((col < 0) or (col > len(self._header) - 1), edata)
            else:
                hnf_ex(
                    (isinstance(col, int) and ((col < 0) or (col > self._data_cols)))
                    or (
                        isinstance(col, str) and (col.upper() not in self._header_upper)
                    ),
                    edata,
                )
        return col_list