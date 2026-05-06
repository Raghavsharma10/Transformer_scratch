def warning(self, s):
        """
        Prints out a warning message to stderr.
        :param s: The warning string to print
        :return: None
        """
        print("   WARNING: '%s', %s" % (self.src_id, s), file=sys.stderr)