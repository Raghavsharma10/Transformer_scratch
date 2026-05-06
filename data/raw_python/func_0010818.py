def derive_fields(self):
        """
        Default implementation
        """
        fields = []
        if self.fields:
            fields.append(self.fields)

        return fields