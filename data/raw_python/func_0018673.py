def extend_schema_spec(self) -> None:
        """ Injects the block start and end times """
        super().extend_schema_spec()

        if self.ATTRIBUTE_FIELDS in self._spec:
            # Add new fields to the schema spec. Since `_identity` is added by the super, new elements are added after
            predefined_field = self._build_time_fields_spec(self._spec[self.ATTRIBUTE_NAME])
            self._spec[self.ATTRIBUTE_FIELDS][1:1] = predefined_field

            # Add new field schema to the schema loader
            for field_schema in predefined_field:
                self.schema_loader.add_schema_spec(field_schema, self.fully_qualified_name)