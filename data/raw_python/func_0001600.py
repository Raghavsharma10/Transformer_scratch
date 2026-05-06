def apply_defaults(self, instance):
        """Applies the defaults described by the this schema to the given
        document instance as appropriate. Defaults are only applied to
        fields which are currently unset."""
        for field, spec in self.doc_spec.iteritems():
            field_type = spec['type']
            if field not in instance:
                if 'default' in spec:
                    default = spec['default']
                    if callable(default):
                        instance[field] = default()
                    else:
                        instance[field] = copy.deepcopy(default)
            # Determine if a value already exists for the field
            if field in instance:
                value = instance[field]

                # recurse into nested docs
                if isinstance(field_type, Schema) and isinstance(value, dict):
                    field_type.apply_defaults(value)

                elif isinstance(field_type, Array) and isinstance(field_type.contained_type, Schema) and isinstance(value, list):
                    for item in value:
                        field_type.contained_type.apply_defaults(item)