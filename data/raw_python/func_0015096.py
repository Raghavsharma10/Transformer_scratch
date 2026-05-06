def _check_required(self, requires, **kwargs):
        """
        Checks kwargs for the values specified in 'requires', which is a tuple
        of strings. These strings are the NVP names of the required values.
        """
        for req in requires:
            # PayPal api is never mixed-case.
            if req.lower() not in kwargs and req.upper() not in kwargs:
                raise PayPalError('missing required : %s' % req)