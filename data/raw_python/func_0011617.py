def get_catalogue_header_value(cls, catalog, key):
        """Get `.po` header value."""
        header_value = None
        if '' in catalog:
            for line in catalog[''].split('\n'):
                if line.startswith('%s:' % key):
                    header_value = line.split(':', 1)[1].strip()

        return header_value