def hook_changed(self, hook, new_data):
        """Called whenever the data for a hook changed."""
        for field in self.hooks[hook]:
            widget = self.widgets[field]
            field.hook_changed(hook, widget, new_data)