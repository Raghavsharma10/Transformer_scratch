def check_arguments(self):
        """Sanity check the arguments passed in.

        Uses the boolean functions specified in the subclasses in the
        _valid_arguments dictionary to determine if an argument is valid
        or invalid.
        """
        for k, v in self.Parameters.iteritems():
            if self.Parameters[k].isOn():
                if k in self._valid_arguments:
                    if not self._valid_arguments[k](v.Value):
                        error_message = 'Invalid argument (%s) ' % v.Value
                        error_message += 'for parameter %s\n' % k
                        raise InvalidArgumentApplicationError(error_message)