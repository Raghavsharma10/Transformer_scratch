def next(self):
        """Get the next processable item of the queue.

        A processable item is supposed to have the status `queued`.

        Returns:
            None : If no key is found.
            Int: If a valid entry is found.

        """
        smallest = None
        for key in self.queue.keys():
            if self.queue[key]['status'] == 'queued':
                if smallest is None or key < smallest:
                    smallest = key
        return smallest