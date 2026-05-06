def remove(self):
        """
        Removes an item from ListNode.

        Raises:
            TypeError: If it's called on container ListNode (intstead of ListNode's item)

        Note:
            Parent object should be explicitly saved.
        """
        if not self._is_item:
            raise TypeError("Should be called on an item, not ListNode's itself.")
        self.container.node_stack.remove(self)