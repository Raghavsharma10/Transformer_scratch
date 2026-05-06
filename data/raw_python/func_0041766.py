def get_command_line(self):
    """Returns the command line for the job."""
    # In python 2, the command line is unicode, which needs to be converted to string before pickling;
    # In python 3, the command line is bytes, which can be pickled directly
    return loads(self.command_line) if isinstance(self.command_line, bytes) else loads(self.command_line.encode())