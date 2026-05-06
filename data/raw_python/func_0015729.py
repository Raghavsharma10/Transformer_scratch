def _create_enum_class(ffi, type_name, prefix, flags=False):
    """Returns a new shiny class for the given enum type"""

    class _template(int):
        _map = {}

        @property
        def value(self):
            return int(self)

        def __str__(self):
            return self._map.get(self, "Unknown")

        def __repr__(self):
            return "%s.%s" % (type(self).__name__, str(self))

    class _template_flags(int):
        _map = {}

        @property
        def value(self):
            return int(self)

        def __str__(self):
            names = []
            val = int(self)
            for flag, name in self._map.items():
                if val & flag:
                    names.append(name)
                    val &= ~flag
            if val:
                names.append(str(val))
            return " | ".join(sorted(names or ["Unknown"]))

        def __repr__(self):
            return "%s(%s)" % (type(self).__name__, str(self))

    if flags:
        template = _template_flags
    else:
        template = _template

    cls = type(type_name, template.__bases__, dict(template.__dict__))
    prefix_len = len(prefix)
    for value, name in ffi.typeof(type_name).elements.items():
        assert name[:prefix_len] == prefix
        name = name[prefix_len:]
        setattr(cls, name, cls(value))
        cls._map[value] = name

    return cls