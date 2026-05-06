def get_compression_extension(self):
        """
        Find the filename extension for the 'docker save' output, which
        may or may not be compressed.

        Raises OsbsValidationException if the extension cannot be
        determined due to a configuration error.

        :returns: str including leading dot, or else None if no compression
        """

        build_request = BuildRequest(build_json_store=self.os_conf.get_build_json_store())
        inner = build_request.inner_template
        postbuild_plugins = inner.get('postbuild_plugins', [])
        for plugin in postbuild_plugins:
            if plugin.get('name') == 'compress':
                args = plugin.get('args', {})
                method = args.get('method', 'gzip')
                if method == 'gzip':
                    return '.gz'
                elif method == 'lzma':
                    return '.xz'
                raise OsbsValidationException("unknown compression method '%s'"
                                              % method)

        return None