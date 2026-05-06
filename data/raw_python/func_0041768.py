def get_array(self):
    """Returns the array arguments for the job; usually a string."""
    # In python 2, the command line is unicode, which needs to be converted to string before pickling;
    # In python 3, the command line is bytes, which can be pickled directly
    return loads(self.array_string) if isinstance(self.array_string, bytes) else loads(self.array_string.encode())