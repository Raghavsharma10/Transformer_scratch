def clean(self):
        """Remove all temporary files."""

        rnftools.utils.shell('rm -fR "{}" "{}"'.format(self.report_dir, self._html_fn))