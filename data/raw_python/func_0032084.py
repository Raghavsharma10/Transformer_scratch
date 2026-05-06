def add_item(self, item):
        """Adds an item to the batch."""

        if not isinstance(item, JsonRpcResponse):
            raise TypeError(
                "Expected JsonRpcResponse but got {} instead".format(type(item).__name__))

        self.items.append(item)