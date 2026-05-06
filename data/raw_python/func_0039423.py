def put(self, stream, cmd):
        """ Spawn a new background process """

        if len(self.q) < self.max_size:
            if stream['id'] in self.q:
                raise QueueDuplicate
            p = self.call(stream, cmd)
            self.q[stream['id']] = p
        else:
            raise QueueFull