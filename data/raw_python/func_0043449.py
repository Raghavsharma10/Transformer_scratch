def copy(self):
        """
        Creates a copy of model
        """
        return self.__class__(field_type=self.get_field_type(), data=self.export_data())