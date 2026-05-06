def show(self, filter=None):
        """ Print the list of commands currently in the queue. If filter is
        given, print only commands that match the filter.
        """
        for command in self._commands:
            if command[0] is None:  # or command[1] in self._invalid_objects:
                continue  # Skip nill commands
            if filter and command[0] != filter:
                continue
            t = []
            for e in command:
                if isinstance(e, np.ndarray):
                    t.append('array %s' % str(e.shape))
                elif isinstance(e, str):
                    s = e.strip()
                    if len(s) > 20:
                        s = s[:18] + '... %i lines' % (e.count('\n')+1)
                    t.append(s)
                else:
                    t.append(e)
            print(tuple(t))