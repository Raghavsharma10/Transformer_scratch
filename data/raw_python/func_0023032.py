def add_number_widget(self, ref, x=1, value=1):
        """ Add Number Widget """

        if ref not in self.widgets:
            widget = widgets.NumberWidget(screen=self, ref=ref, x=x, value=value)
            self.widgets[ref] = widget
            return self.widgets[ref]