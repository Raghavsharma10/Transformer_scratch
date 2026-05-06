def _emit_error(cls, message):
        """Print an error message to STDERR."""
        sys.stderr.write('ERROR: {message}\n'.format(message=message))
        sys.stderr.flush()