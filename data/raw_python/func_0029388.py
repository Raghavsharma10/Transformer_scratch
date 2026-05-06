def show(self, **kwargs):
        """Shows the menu. Any `kwargs` supplied will be passed to
        `show_menu()`."""
        show_kwargs = copy.deepcopy(self._show_kwargs)
        show_kwargs.update(kwargs)
        return show_menu(self.entries, **show_kwargs)