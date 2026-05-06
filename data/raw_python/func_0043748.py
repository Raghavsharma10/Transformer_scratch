def link_attrs(self, func):
        """
        Add a callback for adding attributes to links.

        The callback should take a single argument, the url, and
        should return additional text to be inserted in the link tag,
        i.e. ``"target="_blank"``.

        You can use this method as a decorator on the function you
        want to set as the callback.
        """
        @libmarkdown.e_flags_callback
        def _link_attrs_func(string, size, context):
            ret = func(string[:size])
            if ret is not None:
                buf = ctypes.create_string_buffer(ret)
                self._alloc.append(buf)
                return ctypes.addressof(buf)

        self._link_attrs_func = _link_attrs_func
        return func