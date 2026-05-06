def make_field_objects(field_data, names):
        # type: (List[Dict[Text, Text]], Names) -> List[Field]
        """We're going to need to make message parameters too."""
        field_objects = []
        field_names = []  # type: List[Text]
        for field in field_data:
            if hasattr(field, 'get') and callable(field.get):
                atype = cast(Text, field.get('type'))
                name = cast(Text, field.get('name'))

                # null values can have a default value of None
                has_default = False
                default = None
                if 'default' in field:
                    has_default = True
                    default = field.get('default')

                order = field.get('order')
                doc = field.get('doc')
                other_props = get_other_props(field, FIELD_RESERVED_PROPS)
                new_field = Field(atype, name, has_default, default, order, names, doc,
                                 other_props)
                # make sure field name has not been used yet
                if new_field.name in field_names:
                    fail_msg = 'Field name %s already in use.' % new_field.name
                    raise SchemaParseException(fail_msg)
                field_names.append(new_field.name)
            else:
                raise SchemaParseException('Not a valid field: %s' % field)
            field_objects.append(new_field)
        return field_objects