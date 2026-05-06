def load(self, filename):
        """
        Read a file of known SSH host keys, in the format used by openssh.
        This type of file unfortunately doesn't exist on Windows, but on
        posix, it will usually be stored in
        C{os.path.expanduser("~/.ssh/known_hosts")}.

        If this method is called multiple times, the host keys are merged,
        not cleared.  So multiple calls to C{load} will just call L{add},
        replacing any existing entries and adding new ones.

        @param filename: name of the file to read host keys from
        @type filename: str

        @raise IOError: if there was an error reading the file
        """
        f = open(filename, 'r')
        for line in f:
            line = line.strip()
            if (len(line) == 0) or (line[0] == '#'):
                continue
            e = HostKeyEntry.from_line(line)
            if e is not None:
                self._entries.append(e)
        f.close()