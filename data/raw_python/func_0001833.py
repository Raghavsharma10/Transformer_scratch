def get_range(self, i):
        """
        Check and retrieve range if value is a valid range.

        Here we are looking to see if the value is series or range.
        We look for `{1..2[..inc]}` or `{a..z[..inc]}` (negative numbers are fine).
        """

        try:
            m = i.match(RE_INT_ITER)
            if m:
                return self.get_int_range(*m.groups())

            m = i.match(RE_CHR_ITER)
            if m:
                return self.get_char_range(*m.groups())
        except Exception:  # pragma: no cover
            # TODO: We really should never fail here,
            # but if we do, assume the sequence range
            # was invalid. This catch can probably
            # be removed in the future with more testing.
            pass

        return None