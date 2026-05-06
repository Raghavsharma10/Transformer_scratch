def set(self, key=None, value=None, **kwargs):
        """
        `set` is a way to add raw properties to the request,
        for features that this module does not
        support or supports incompletely. For convenience's
        sake, it will serialize Column objects but will
        leave any other kind of value alone.
        """

        serialize = partial(self.api.columns.serialize, greedy=False)

        if key and value:
            self.raw[key] = serialize(value)
        elif key or kwargs:
            properties = key or kwargs
            for key, value in properties.items():
                self.raw[key] = serialize(value)
        else:
            raise ValueError(
                "Query#set requires a key and value, a properties dictionary or keyword arguments.")

        return self