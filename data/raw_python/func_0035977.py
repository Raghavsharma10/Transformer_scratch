def _emit_message(cls, message):
        """Print a message to STDOUT."""
        sys.stdout.write(message)
        sys.stdout.flush()