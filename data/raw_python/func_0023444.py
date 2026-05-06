def switch(self, first, second):
        """Switch two entries in the queue. Return False if an entry doesn't exist."""
        allowed_states = ['queued', 'stashed']
        if first in self.queue and second in self.queue \
                and self.queue[first]['status'] in allowed_states\
                and self.queue[second]['status'] in allowed_states:

            tmp = self.queue[second].copy()
            self.queue[second] = self.queue[first].copy()
            self.queue[first] = tmp
            self.write()
            return True
        return False