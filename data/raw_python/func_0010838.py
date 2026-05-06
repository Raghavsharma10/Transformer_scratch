def derive_fields(self):
        """
        Derives our fields.
        """
        if self.fields is not None:
            fields = list(self.fields)
        else:
            form = self.form
            fields = []
            for field in form:
                fields.append(field.name)

            # this is slightly confusing but we add in readonly fields here because they will still
            # need to be displayed
            readonly = self.derive_readonly()
            if readonly:
                fields += readonly

        # remove any excluded fields
        for exclude in self.derive_exclude():
            if exclude in fields:
                fields.remove(exclude)

        return fields