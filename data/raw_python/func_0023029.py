def add_hbar_widget(self, ref, x=1, y=1, length=10):
        """ Add Horizontal Bar Widget """

        if ref not in self.widgets:
            widget = widgets.HBarWidget(screen=self, ref=ref, x=x, y=y, length=length)
            self.widgets[ref] = widget
            return self.widgets[ref]