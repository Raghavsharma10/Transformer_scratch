def _set(self, name, value):
        "Proxy to set a property of the widget element."
        return self.widget(self.widget_element._set(name, value))