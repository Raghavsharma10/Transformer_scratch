def add_title_widget(self, ref, text="Title"):
        """ Add Title Widget """

        if ref not in self.widgets:
            widget = widgets.TitleWidget(screen=self, ref=ref, text=text)
            self.widgets[ref] = widget
            return self.widgets[ref]