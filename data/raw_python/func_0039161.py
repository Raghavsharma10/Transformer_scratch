def parse_dict(self, attrs):
        """Read a dict to attributes."""
        attrs = attrs or {}
        ident = attrs.get("id", "")
        classes = attrs.get("classes", [])
        kvs = OrderedDict((k, v) for k, v in attrs.items()
                          if k not in ("classes", "id"))

        return ident, classes, kvs