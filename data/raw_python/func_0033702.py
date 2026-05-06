def _format_function_arguments(self, opts):
        """Format a series of function arguments in a Mothur script."""
        params = [self.Parameters[x] for x in opts]
        return ', '.join(filter(None, map(str, params)))