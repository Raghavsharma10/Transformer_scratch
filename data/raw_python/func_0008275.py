def _fill_from_meta(self, Meta: Type[object], base_classes_meta, mcs_args: McsArgs):
        """
        Iterate over our supported meta options, and set attributes on the factory
        instance (self) for each meta option's name/value. Raises ``TypeError`` if
        we discover any unsupported meta options on the class-under-construction's
        ``class Meta``.
        """
        # Exclude private/protected fields from the Meta
        meta_attrs = {} if not Meta else {k: v for k, v in vars(Meta).items()
                                          if not k.startswith('_')}

        for option in self._get_meta_options():
            existing = getattr(self, option.name, None)
            if existing and not (existing in self._allowed_properties
                                 and not isinstance(existing, property)):
                raise RuntimeError("Can't override field {name}."
                                   "".format(name=option.name))
            value = option.get_value(Meta, base_classes_meta, mcs_args)
            option.check_value(value, mcs_args)
            meta_attrs.pop(option.name, None)
            if option.name != '_':
                setattr(self, option.name, value)

        if meta_attrs:
            # Only allow attributes on the Meta that have a respective MetaOption
            raise TypeError(
                '`class Meta` for {cls} got unknown attribute(s) {attrs}'.format(
                    cls=mcs_args.name,
                    attrs=', '.join(sorted(meta_attrs.keys()))))