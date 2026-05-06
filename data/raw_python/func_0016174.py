def add_name(self, name_attr, space_attr, new_schema):
        # type: (Text, Optional[Text], NamedSchema) -> Name
        """
        Add a new schema object to the name set.

          @arg name_attr: name value read in schema
          @arg space_attr: namespace value read in schema.

          @return: the Name that was just added.
        """
        to_add = Name(name_attr, space_attr, self.default_namespace)

        if to_add.fullname in VALID_TYPES:
            fail_msg = '%s is a reserved type name.' % to_add.fullname
            raise SchemaParseException(fail_msg)
        elif to_add.fullname in self.names:
            fail_msg = 'The name "%s" is already in use.' % to_add.fullname
            raise SchemaParseException(fail_msg)

        self.names[to_add.fullname] = new_schema
        return to_add