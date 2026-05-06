def create_graphics(self):
        """Create images related to this BAM file using GnuPlot."""

        rnftools.utils.shell('"{}" "{}"'.format("gnuplot", self._gp_fn))

        if self.render_pdf_method is not None:
            svg_fn = self._svg_fn
            pdf_fn = self._pdf_fn
            svg42pdf(svg_fn, pdf_fn, method=self.render_pdf_method)