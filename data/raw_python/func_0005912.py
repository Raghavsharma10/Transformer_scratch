def _set_name(self, version=AUTO):
        """Returns plugin name."""

        name = 'python'

        if version:
            if version is AUTO:
                version = sys.version_info[0]

                if version == 2:
                    version = ''

            name = '%s%s' % (name, version)

        self.name = name