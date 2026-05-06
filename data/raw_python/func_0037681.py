def deconstruct(self):
        """
        to support Django 1.7 migrations, see also the add_introspection_rules
        section at bottom of this file for South + earlier Django versions
        """
        name, path, args, kwargs = super(
            ExclusiveBooleanField, self).deconstruct()
        if self._on_fields:
            kwargs['on'] = self._on_fields
        return name, path, args, kwargs