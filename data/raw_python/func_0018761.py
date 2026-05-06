def extend_schema_spec(self) -> None:
        """ Injects the identity field """
        super().extend_schema_spec()

        identity_field = {
            'Name': '_identity',
            'Type': BtsType.STRING,
            'Value': 'identity',
            ATTRIBUTE_INTERNAL: True
        }

        if self.ATTRIBUTE_FIELDS in self._spec:
            self._spec[self.ATTRIBUTE_FIELDS].insert(0, identity_field)
            self.schema_loader.add_schema_spec(identity_field, self.fully_qualified_name)