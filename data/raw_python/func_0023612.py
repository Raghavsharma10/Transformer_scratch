def random_connection(self):
        '''Pick a random living connection'''
        # While at the moment there's no need for this to be a context manager
        # per se, I would like to use that interface since I anticipate
        # adding some wrapping around it at some point.
        yield random.choice(
            [conn for conn in self.connections() if conn.alive()])