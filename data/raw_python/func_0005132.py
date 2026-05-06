def _format_attrs(self):
        """ Formats the self.attrs #OrderedDict """
        _bold = bold
        _colorize = colorize
        if not self.pretty:
            _bold = lambda x: x
            _colorize = lambda x, c: x
        attrs = []
        add_attr = attrs.append
        if self.doc and hasattr(self.obj, "__doc__"):
            # Optionally attaches documentation
            if self.obj.__doc__:
                add_attr("`{}`".format(self.obj.__doc__.strip()))
        if self.attrs:
            # Attach request attributes
            for key, value in self.attrs.items():
                value, color = value
                try:
                    value = value or \
                        self._getattrs(getattr, self.obj, key.split("."))
                except AttributeError:
                    pass
                value = _colorize(value, color) if color else value
                v = None
                if value is not None:
                    value = "`{}`".format(value) \
                        if isinstance(value, Look.str_) else value
                    k, v = _bold(key), value
                else:
                    k, v = _bold(key), str(value)
                if v:
                    k = '{}='.format(k) if not self._no_keys else ''
                    add_attr("{}{}".format(k, v))
        if len(attrs):
            breaker = "\n    " if self.line_break and len(attrs) > 1 else ""
            return breaker + ((", "+breaker).join(attrs)) + breaker.strip(" ")
        else:
            return ""