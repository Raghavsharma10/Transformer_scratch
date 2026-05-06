def log_tail(self, nlines=10):
        """
        Return the last ``nlines`` lines of the log file
        """
        log_path = os.path.join(self.working_dir, self.log_name)
        with open(log_path) as fp:
            d = collections.deque(maxlen=nlines)
            d.extend(fp)
            return ''.join(d)