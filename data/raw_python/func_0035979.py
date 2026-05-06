def _emit_warning(cls, message):
        """Print an warning message to STDERR."""
        sys.stderr.write('WARNING: {message}\n'.format(message=message))
        sys.stderr.flush()