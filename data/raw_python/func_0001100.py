def shouldRollover(self, record):
        """Determine if rollover should occur.

        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if self.maxBytes > 0:  # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  # due to non-posix-compliant win feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        return 0