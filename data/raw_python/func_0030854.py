def get_heatmap(self, highlight_genes=None, highlight_samples=None,
                          highlight_color=None, **kwargs):
        """Generate a heatmap (`ExpHeatmap`) of the matrix.

        See :class:`ExpHeatmap` constructor for keyword arguments.

        Parameters
        ----------
        highlight_genes : list of str
            List of genes to highlight
        highlight_color : str
            Color to use for highlighting

        Returns
        -------
        `ExpHeatmap`
            The heatmap.
        """
        from .visualize import ExpHeatmap
        from .visualize import HeatmapGeneAnnotation
        from .visualize import HeatmapSampleAnnotation

        #if highlight_genes is not None:
        #    assert isinstance(highlight_genes, Iterable)
        #if highlight_samples is not None:
        #    assert isinstance(highlight_genes, Iterable)
        if highlight_color is not None:
            assert isinstance(highlight_color, str)

        if highlight_color is None:
            highlight_color = 'blue'

        if highlight_genes is None:
            highlight_genes = []

        if highlight_samples is None:
            highlight_samples = []

        gene_annotations = kwargs.pop('gene_annotations', [])
        for g in highlight_genes:
            gene_annotations.append(
                HeatmapGeneAnnotation(g, highlight_color, label=g))

        sample_annotations = kwargs.pop('sample_annotations', [])
        for s in highlight_samples:
            sample_annotations.append(
                HeatmapSampleAnnotation(s, highlight_color, label=s)
            )

        return ExpHeatmap(self,
                          gene_annotations=gene_annotations,
                          sample_annotations=sample_annotations,
                          **kwargs)