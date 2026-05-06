def copy_with(self, **kwargs):
        """Return a copy with (a few) changed attributes

           The keyword arguments are the attributes to be replaced by new
           values. All other attributes are copied (or referenced) from the
           original object. This only works if the constructor takes all
           (read-only) attributes as arguments.
        """
        attrs = {}
        for key, descriptor in self.__class__.__dict__.items():
            if isinstance(descriptor, ReadOnlyAttribute):
                attrs[key] = descriptor.__get__(self)
        for key in kwargs:
            if key not in attrs:
                raise TypeError("Unknown attribute: %s" % key)
        attrs.update(kwargs)
        return self.__class__(**attrs)