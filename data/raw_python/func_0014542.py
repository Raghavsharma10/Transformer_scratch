def to_line(self):
        """
        Returns a string in OpenSSH known_hosts file format, or None if
        the object is not in a valid state.  A trailing newline is
        included.
        """
        if self.valid:
            return '%s %s %s\n' % (','.join(self.hostnames), self.key.get_name(),
                   self.key.get_base64())
        return None