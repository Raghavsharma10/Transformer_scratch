def get_exec_dir(self):
    """Returns the command line for the job."""
    # In python 2, the command line is unicode, which needs to be converted to string before pickling;
    # In python 3, the command line is bytes, which can be pickled directly
    return str(os.path.realpath(self.exec_dir)) if self.exec_dir is not None else None