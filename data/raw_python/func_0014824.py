def deconstruct(self):
        """Handle django.db.migrations."""
        name, path, args, kwargs = super(VersionField, self).deconstruct()
        kwargs['partial'] = self.partial
        kwargs['coerce'] = self.coerce
        return name, path, args, kwargs