def set_value(self, obj, value):
        """Sets value to model if not empty"""
        if value:
            obj.set_field_value(self.name, value)
        else:
            self.delete_value(obj)