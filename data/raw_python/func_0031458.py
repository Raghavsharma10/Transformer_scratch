def get_static_enrichment(
            self, genes: Iterable[str],
            pval_thresh: float,
            adjust_pval_thresh: bool = True,
            K_min: int = 3,
            gene_set_ids: Iterable[str] = None) -> StaticGSEResult:
        """Find enriched gene sets in a set of genes.

        Parameters
        ----------
        genes : set of str
            The set of genes to test for gene set enrichment.
        pval_thresh : float
            The significance level (p-value threshold) to use in the analysis.
        adjust_pval_thresh : bool, optional
            Whether to adjust the p-value threshold using a Bonferroni
            correction. (Warning: This is a very conservative correction!)
            [True]
        K_min : int, optional
            The minimum number of gene set genes present in the analysis. [3]
        gene_set_ids : Iterable or None
            A list of gene set IDs to test. If ``None``, all gene sets are
            tested that meet the :attr:`K_min` criterion.

        Returns
        -------
        list of `StaticGSEResult`
            A list of all significantly enriched gene sets. 
        """
        genes = set(genes)
        gene_set_coll = self._gene_set_coll
        gene_sets = self._gene_set_coll.gene_sets
        gene_memberships = self._gene_memberships
        sorted_genes = sorted(genes)

        # test only some terms?
        if gene_set_ids is not None:
            gs_indices = np.int64([self._gene_set_coll.index(id_)
                                   for id_ in gene_set_ids])
            gene_sets = [gene_set_coll[id_] for id_ in gene_set_ids]
            # gene_set_coll = GeneSetCollection(gene_sets)
            gene_memberships = gene_memberships[:, gs_indices]  # not a view!

        # determine K's
        K_vec = np.sum(gene_memberships, axis=0, dtype=np.int64)

        # exclude terms with too few genes
        sel = np.nonzero(K_vec >= K_min)[0]
        K_vec = K_vec[sel]
        gene_sets = [gene_sets[j] for j in sel]
        gene_memberships = gene_memberships[:, sel]

        # determine k's, ignoring unknown genes
        unknown = 0
        sel = []
        filtered_genes = []
        logger.debug('Looking up indices for %d genes...', len(sorted_genes))
        for i, g in enumerate(sorted_genes):
            try:
                idx = self._gene_indices[g]
            except KeyError:
                unknown += 1
            else:
                sel.append(idx)
                filtered_genes.append(g)

        sel = np.int64(sel)
        gene_indices = np.int64(sel)
        # gene_memberships = gene_memberships[sel, :]
        k_vec = np.sum(gene_memberships[sel, :], axis=0, dtype=np.int64)
        if unknown > 0:
            logger.warn('%d / %d unknown genes (%.1f %%), will be ignored.',
                        unknown, len(genes),
                        100 * (unknown / float(len(genes))))

        # determine n and N
        n = len(filtered_genes)
        N, m = gene_memberships.shape
        logger.info('Conducting %d tests.', m)

        # correct p-value threshold, if specified
        final_pval_thresh = pval_thresh
        if adjust_pval_thresh:
            final_pval_thresh /= float(m)
            logger.info('Using Bonferroni-corrected p-value threshold: %.1e',
                        final_pval_thresh)

        # calculate p-values and get significantly enriched gene sets
        enriched = []

        logger.debug('N=%d, n=%d', N, n)
        sys.stdout.flush()
        genes = self._valid_genes
        for j in range(m):
            pval = hypergeom.sf(k_vec[j] - 1, N, K_vec[j], n)
            if pval <= final_pval_thresh:
                # found significant enrichment
                # sel_genes = [filtered_genes[i] for i in np.nonzero(gene_memberships[:, j])[0]]
                sel_genes = [genes[i] for i in
                             np.nonzero(gene_memberships[gene_indices, j])[0]]
                enriched.append(
                    StaticGSEResult(gene_sets[j], N, n, set(sel_genes), pval))

        return enriched