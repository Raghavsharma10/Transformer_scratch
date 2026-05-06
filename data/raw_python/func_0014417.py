def get_todo_items(self, **kwargs):
        '''
        Returns an iterator that will provide each item in the todo queue. Note that to complete each item you have to run complete method with the output of this iterator.

        That will move the item to the done directory and prevent it from being retrieved in the future.
        '''
        def inner(self):
            for item in self.get_all_as_list():
                yield item
            self._unlock()

        if not self._is_locked():
            if self._lock():
                return inner(self)
        raise RuntimeError("RuntimeError: Index Already Locked")