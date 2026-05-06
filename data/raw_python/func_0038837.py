def choices(cls, value_field='canonical_name', display_field='display_name'):
        """
        DEPRECATED

        Returns a list of 2-tuples to be used as an argument to Django Field.choices

        Implementation note: choices() can't be a property
        See:
            http://www.no-ack.org/2011/03/strange-behavior-with-properties-on.html
            http://utcc.utoronto.ca/~cks/space/blog/python/UsingMetaclass03
        """
        return [m.choicify(value_field=value_field, display_field=display_field) for m in cls.members()]