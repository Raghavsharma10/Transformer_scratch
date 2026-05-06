def grow(self):
        "Add another worker to the pool."
        t = self.worker_factory(self)
        t.start()
        self._size += 1