def _extract_cause(cls, exc_val):
        """Helper routine to extract nested cause (if any)."""
        # See: https://www.python.org/dev/peps/pep-3134/ for why/what
        # these are...
        #
        # '__cause__' attribute for explicitly chained exceptions
        # '__context__' attribute for implicitly chained exceptions
        # '__traceback__' attribute for the traceback
        #
        # See: https://www.python.org/dev/peps/pep-0415/ for why/what
        # the '__suppress_context__' is/means/implies...
        nested_exc_vals = []
        seen = [exc_val]
        while True:
            suppress_context = getattr(
                exc_val, '__suppress_context__', False)
            if suppress_context:
                attr_lookups = ['__cause__']
            else:
                attr_lookups = ['__cause__', '__context__']
            nested_exc_val = None
            for attr_name in attr_lookups:
                attr_val = getattr(exc_val, attr_name, None)
                if attr_val is None:
                    continue
                nested_exc_val = attr_val
            if nested_exc_val is None or nested_exc_val in seen:
                break
            seen.append(nested_exc_val)
            nested_exc_vals.append(nested_exc_val)
            exc_val = nested_exc_val
        last_cause = None
        for exc_val in reversed(nested_exc_vals):
            f = cls.from_exception(exc_val, cause=last_cause,
                                   find_cause=False)
            last_cause = f
        return last_cause