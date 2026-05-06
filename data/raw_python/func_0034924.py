def encode(self, obj):
        """ Add the given object to the result.
        """
        if isinstance(obj, int_like_types):
            self.result.append("i%de" % obj)
        elif isinstance(obj, string_types):
            self.result.extend([str(len(obj)), ':', str(obj)])
        elif hasattr(obj, "__bencode__"):
            self.encode(obj.__bencode__())
        elif hasattr(obj, "items"):
            # Dictionary
            self.result.append('d')
            for key, val in sorted(obj.items()):
                key = str(key)
                self.result.extend([str(len(key)), ':', key])
                self.encode(val)
            self.result.append('e')
        else:
            # Treat as iterable
            try:
                items = iter(obj)
            except TypeError as exc:
                raise BencodeError("Unsupported non-iterable object %r of type %s (%s)" % (
                    obj, type(obj), exc
                ))
            else:
                self.result.append('l')
                for item in items:
                    self.encode(item)
                self.result.append('e')

        return self.result