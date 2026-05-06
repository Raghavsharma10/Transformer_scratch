def derive_fields(self):
        """
        Derives our fields.  We first default to using our 'fields' variable if available,
        otherwise we figure it out from our object.
        """
        if self.fields:
            return list(self.fields)

        else:
            fields = []
            for field in self.object._meta.fields:
                fields.append(field.name)

            # only exclude?  then remove those items there
            exclude = self.derive_exclude()

            # remove any excluded fields
            fields = [field for field in fields if field not in exclude]

            return fields