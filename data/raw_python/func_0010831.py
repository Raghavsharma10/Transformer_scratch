def derive_fields(self):
        """
        Derives our fields.
        """
        if self.fields:
            return self.fields

        else:
            fields = []
            for field in self.object_list.model._meta.fields:
                if field.name != 'id':
                    fields.append(field.name)
            return fields