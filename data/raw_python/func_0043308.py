def register_child(self, child):
        """
        Register a new child that will be closed whenever the current instance
        closes.

        :param child: The child instance.
        """
        if self.closing:
            child.close()
        else:
            self._children.add(child)
            child.on_closed.connect(self.unregister_child)