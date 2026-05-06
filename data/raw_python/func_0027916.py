def generate(self, field_name, field):
        """Tries to lookup a matching formfield generator (lowercase
        field-classname) and raises a NotImplementedError of no generator
        can be found.
        """

        if hasattr(self, 'generate_%s' % field.__class__.__name__.lower()):
            generator = getattr(
                self,
                'generate_%s' % field.__class__.__name__.lower())
            return generator(
                field_name,
                field,
                (field.verbose_name or field_name).capitalize())
        else:
            raise NotImplementedError('%s is not supported by MongoForm' % \
                field.__class__.__name__)