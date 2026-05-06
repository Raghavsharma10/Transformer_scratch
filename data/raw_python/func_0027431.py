def output(self, _in, out, **kwargs):
        """Wrap translation in Angular module."""
        out.write(
            'angular.module("{0}", ["gettext"]).run('
            '["gettextCatalog", function (gettextCatalog) {{'.format(
                self.catalog_name
            )
        )
        out.write(_in.read())
        out.write('}]);')