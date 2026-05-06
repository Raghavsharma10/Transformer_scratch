def restore(self):
        """
        Destroy all inspectors in exp_list and SinonMock itself
        """
        for expectation in self.exp_list:
            try:
                expectation.restore()
            except ReferenceError:
                pass #ignore removed expectation
        self._queue.remove(self)