def rewrite_links(self, func):
        """
        Add a callback for rewriting links.

        The callback should take a single argument, the url, and
        should return a replacement url.  The callback function is
        called everytime a ``[]()`` or ``<link>`` is processed.

        You can use this method as a decorator on the function you
        want to set as the callback.
        """
        @libmarkdown.e_url_callback
        def _rewrite_links_func(string, size, context):
            ret = func(string[:size])
            if ret is not None:
                buf = ctypes.create_string_buffer(ret)
                self._alloc.append(buf)
                return ctypes.addressof(buf)

        self._rewrite_links_func = _rewrite_links_func
        return func