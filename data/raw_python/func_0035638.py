def south_field_triple(self):
        """Returns a suitable description of this field for South."""
        # We'll just introspect the _actual_ field.
        from south.modelsinspector import introspector
        field_class = '%s.%s' % (self.__module__, self.__class__.__name__)
        args, kwargs = introspector(self)
        kwargs.update({
            'length': repr(self.length),
            'exclude_upper': repr(self.exclude_upper),
            'exclude_lower': repr(self.exclude_lower),
            'exclude_digits': repr(self.exclude_digits),
            'exclude_vowels': repr(self.exclude_vowels),
        })
        # That's our definition!
        return (field_class, args, kwargs)