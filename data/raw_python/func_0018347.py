def remove(self, widget):
        """Remove a widget from the window."""
        for i, (wid, _) in enumerate(self._widgets):
            if widget is wid:
                del self._widgets[i]
                return True

        raise ValueError('Widget not in list')