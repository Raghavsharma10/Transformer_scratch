def create_graphics(self):
        """Create images related to this panel."""

        if len(self._svg_fns) > 0:
            rnftools.utils.shell('"{}" "{}"'.format("gnuplot", self._gp_fn))

            if self.render_pdf_method is not None:
                for svg_fn in self._svg_fns:
                    pdf_fn = re.sub(r'\.svg$', r'.pdf', svg_fn)
                    svg42pdf(svg_fn, pdf_fn, method=self.render_pdf_method)