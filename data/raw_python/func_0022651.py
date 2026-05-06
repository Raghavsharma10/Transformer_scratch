def error(self, s):
        """
        Prints out an error message to stderr.
        :param s: The error string to print
        :return: None
        """
        print("   ERROR: '%s', %s" % (self.src_id, s), file=sys.stderr)