def _validate_rfilter(self, rfilter, letter="d"):
        """Validate that all columns in filter are in header."""
        if letter == "d":
            pexdoc.exh.addai(
                "dfilter",
                (
                    (not self._has_header)
                    and any([not isinstance(item, int) for item in rfilter.keys()])
                ),
            )
        else:
            pexdoc.exh.addai(
                "rfilter",
                (
                    (not self._has_header)
                    and any([not isinstance(item, int) for item in rfilter.keys()])
                ),
            )
        for key in rfilter:
            self._in_header(key)
            rfilter[key] = (
                [rfilter[key]] if isinstance(rfilter[key], str) else rfilter[key]
            )