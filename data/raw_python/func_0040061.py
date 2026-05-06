def load(cls, data):
        """Construct a Constant class from it's dict data.

        .. versionadded:: 0.0.2
        """
        if len(data) == 1:
            for key, value in data.items():
                if "__classname__" not in value:  # pragma: no cover
                    raise ValueError
                name = key
                bases = (Constant,)
                attrs = dict()
                for k, v in value.items():
                    if isinstance(v, dict):
                        if "__classname__" in v:
                            attrs[k] = cls.load({k: v})
                        else:
                            attrs[k] = v
                    else:
                        attrs[k] = v
            return type(name, bases, attrs)
        else:  # pragma: no cover
            raise ValueError