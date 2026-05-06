def on(self, val=None):
        """Turns the MixedParameter ON by setting its Value to val

        An attempt to turn the parameter on with value 'False' will result
            in an error, since this is the same as turning the parameter off.

        Turning the MixedParameter ON without a value or with value 'None'
            will let the parameter behave as a flag.
        """
        if val is False:
            raise ParameterError("Turning the ValuedParameter on with value "
                                 "False is the same as turning it off. Use "
                                 "another value.")
        elif self.IsPath:
            self.Value = FilePath(val)
        else:
            self.Value = val