def after_output(command_status):
    """
    Shell sequence to be run after the command output.

    The ``command_status`` should be in the range 0-255.
    """
    if command_status not in range(256):
        raise ValueError("command_status must be an integer in the range 0-255")
    sys.stdout.write(AFTER_OUTPUT.format(command_status=command_status))
    # Flushing is important as the command timing feature maybe based on
    # AFTER_OUTPUT in the future.
    sys.stdout.flush()