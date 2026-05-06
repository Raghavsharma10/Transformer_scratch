def unregister_child(self, child):
        """
        Unregister an existing child that is no longer to be owned by the
        current instance.

        :param child: The child instance.
        """
        self._children.remove(child)
        child.on_closed.disconnect(self.unregister_child)