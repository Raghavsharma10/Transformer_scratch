def _copy_searchtools(self, renderer=None):
        """Copy and patch searchtools

        This uses the included Sphinx version's searchtools, but patches it to
        remove automatic initialization. This is a fork of
        ``sphinx.util.fileutil.copy_asset``
        """
        log.info(bold('copying searchtools... '), nonl=True)

        if sphinx.version_info < (1, 8):
            search_js_file = 'searchtools.js_t'
        else:
            search_js_file = 'searchtools.js'
        path_src = os.path.join(
            package_dir, 'themes', 'basic', 'static', search_js_file
        )
        if os.path.exists(path_src):
            path_dest = os.path.join(self.outdir, '_static', 'searchtools.js')
            if renderer is None:
                # Sphinx 1.4 used the renderer from the existing builder, but
                # the pattern for Sphinx 1.5 is to pass in a renderer separate
                # from the builder. This supports both patterns for future
                # compatibility
                if sphinx.version_info < (1, 5):
                    renderer = self.templates
                else:
                    from sphinx.util.template import SphinxRenderer
                    renderer = SphinxRenderer()
            with codecs.open(path_src, 'r', encoding='utf-8') as h_src:
                with codecs.open(path_dest, 'w', encoding='utf-8') as h_dest:
                    data = h_src.read()
                    data = self.REPLACEMENT_PATTERN.sub(self.REPLACEMENT_TEXT, data)
                    h_dest.write(renderer.render_string(
                        data,
                        self.get_static_readthedocs_context()
                    ))
        else:
            log.warning('Missing {}'.format(search_js_file))
        log.info('done')