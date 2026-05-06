def convert_widgets(self):
        """
        During form initialization, some widgets have to be replaced by a counterpart suitable to
        be rendered the AngularJS way.
        """
        for field in self.base_fields.values():
            try:
                new_widget = field.get_converted_widget()
            except AttributeError:
                pass
            else:
                if new_widget:
                    field.widget = new_widget