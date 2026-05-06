def summarize_provenance(self):
        """Utility function to summarize provenance files for cached items used by a Cohort.

        At the moment, most PROVENANCE files contain details about packages used to
        generate files. However, this function is generic & so it summarizes the contents
        of those files irrespective of their contents.

        Returns
        ----------
        Dict containing summary of provenance items, among all cache dirs used by the Cohort.

        IE if all provenances are identical across all cache dirs, then a single set of
        provenances is returned. Otherwise, if all provenances are not identical, the provenance
        items per cache_dir are returned.

        See also
        ----------
        `?cohorts.Cohort.summarize_provenance_per_cache` which is used to summarize provenance
        for each existing cache_dir.
        """
        provenance_per_cache = self.summarize_provenance_per_cache()
        summary_provenance = None
        num_discrepant = 0
        for cache in provenance_per_cache:
            if not(summary_provenance):
                ## pick arbitrary provenance & call this the "summary" (for now)
                summary_provenance = provenance_per_cache[cache]
                summary_provenance_name = cache
            ## for each cache, check equivalence with summary_provenance
            num_discrepant += compare_provenance(
                provenance_per_cache[cache],
                summary_provenance,
                left_outer_diff = "In %s but not in %s" % (cache, summary_provenance_name),
                right_outer_diff = "In %s but not in %s" % (summary_provenance_name, cache)
            )
        ## compare provenance across cached items
        if num_discrepant == 0:
            prov = summary_provenance ## report summary provenance if exists
        else:
            prov = provenance_per_cache ## otherwise, return provenance per cache
        return(prov)