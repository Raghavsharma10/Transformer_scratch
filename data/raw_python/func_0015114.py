def get_declared_fields(mcs, klass, *args, **kwargs):
        """Updates declared fields with fields converted from the
        Mongoengine model passed as the `model` class Meta option.
        """
        declared_fields = kwargs.get('dict_class', dict)()
        # Generate the fields provided through inheritance
        opts = klass.opts
        model = getattr(opts, 'model', None)
        if model:
            converter = opts.model_converter()
            declared_fields.update(converter.fields_for_model(
                model,
                fields=opts.fields
            ))
        # Generate the fields provided in the current class
        base_fields = super(SchemaMeta, mcs).get_declared_fields(
            klass, *args, **kwargs
        )
        declared_fields.update(base_fields)
        # Customize fields with provided kwargs
        for field_name, field_kwargs in klass.opts.model_fields_kwargs.items():
            field = declared_fields.get(field_name, None)
            if field:
                # Copy to prevent alteration of a possible parent class's field
                field = copy.copy(field)
                for key, value in field_kwargs.items():
                    setattr(field, key, value)
                declared_fields[field_name] = field
        if opts.model_dump_only_pk and opts.model:
            # If primary key is automatically generated (nominal case), we
            # must make sure this field is read-only
            if opts.model._auto_id_field is True:
                field_name = opts.model._meta['id_field']
                id_field = declared_fields.get(field_name)
                if id_field:
                    # Copy to prevent alteration of a possible parent class's field
                    id_field = copy.copy(id_field)
                    id_field.dump_only = True
                    declared_fields[field_name] = id_field
        return declared_fields