def add_view(self, *args, **kwargs):
        """
        Create a new ViewBox and add it as a child widget.

        All arguments are given to ViewBox().
        """
        from .viewbox import ViewBox
        view = ViewBox(*args, **kwargs)
        return self.add_widget(view)