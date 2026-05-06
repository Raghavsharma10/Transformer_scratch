def convert(self, args, handler=None):
        """Prepare filters."""
        name = args
        field = attr = None
        opts = ()
        if isinstance(args, (list, tuple)):
            name, *opts = args
            if opts:
                attr = opts.pop()
            if opts:
                field = opts.pop()

        if not field and handler and handler.Schema:
            field = handler.Schema._declared_fields.get(attr or name) or \
                self.FILTER_CLASS.field_cls()
            field.attribute = field.attribute or attr or name
        return self.FILTER_CLASS(name, attr=attr, field=field, *opts)