def _unquote(self, val):
        """Unquote a value if necessary."""
        if (len(val) >= 2) and (val[0] in ("'", '"')) and (val[0] == val[-1]):
            val = val[1:-1]
        return val