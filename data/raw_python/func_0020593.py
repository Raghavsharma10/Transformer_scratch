def fire(self, event):
        """
        Fires a specified DOM event on the current node.

        :param event: the name of the event to fire (e.g., 'click').

        Returns the :class:`zombie.dom.DOMNode` to allow function chaining.
        """
        self.browser.fire(self.element, event)
        return self