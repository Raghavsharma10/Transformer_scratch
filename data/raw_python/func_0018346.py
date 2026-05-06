def add(self, widget, condition=lambda: 42):
        """
        Add a widget to the widows.

        The widget will auto render. You can use the function like that if you want to keep the widget accecible :
            self.my_widget = self.add(my_widget)
        """

        assert callable(condition)
        assert isinstance(widget, BaseWidget)
        self._widgets.append((widget, condition))

        return widget