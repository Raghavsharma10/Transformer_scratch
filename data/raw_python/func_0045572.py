def _generate_instances(self):
        """
        ListNode item generator. Will be used internally by __iter__ and __getitem__

        Yields:
            ListNode items (instances)
        """
        for node in self.node_stack:
            yield node
        while self._data:
            yield self._make_instance(self._data.pop(0))