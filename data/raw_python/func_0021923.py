def summarize_provenance_per_cache(self):
        """Utility function to summarize provenance files for cached items used by a Cohort,
        for each cache_dir that exists. Only existing cache_dirs are summarized.

        This is a summary of provenance files because the function checks to see whether all
        patients data have the same provenance within the cache dir. The function assumes
        that it will be desireable to have all patients data generated using the same
        environment, for each cache type.

        At the moment, most PROVENANCE files contain details about packages used to generat
        e the cached data file. However, this function is generic & so it summarizes the
        contents of those files irrespective of their contents.

        Returns
        ----------
        Dict containing summarized provenance for each existing cache_dir, after checking
        to see that provenance files are identical among all patients in the data frame for
        that cache_dir.

        If conflicting PROVENANCE files are discovered within a cache-dir:
         - a warning is generated, describing the conflict
         - and, a value of `None` is returned in the dictionary for that cache-dir

        See also
        -----------
        * `?cohorts.Cohort.summarize_provenance` which summarizes provenance files among
        cache_dirs.
        * `?cohorts.Cohort.summarize_dataframe` which hashes/summarizes contents of the data
        frame for this cohort.
        """
        provenance_summary = {}
        df = self.as_dataframe()
        for cache in self.cache_names:
            cache_name = self.cache_names[cache]
            cache_provenance = None
            num_discrepant = 0
            this_cache_dir = path.join(self.cache_dir, cache_name)
            if path.exists(this_cache_dir):
                for patient_id in self._list_patient_ids():
                    patient_cache_dir = path.join(this_cache_dir, patient_id)
                    try:
                        this_provenance = self.load_provenance(patient_cache_dir = patient_cache_dir)
                    except:
                        this_provenance = None
                    if this_provenance:
                        if not(cache_provenance):
                            cache_provenance = this_provenance
                        else:
                            num_discrepant += compare_provenance(this_provenance, cache_provenance)
                if num_discrepant == 0:
                    provenance_summary[cache_name] = cache_provenance
                else:
                    provenance_summary[cache_name] = None
        return(provenance_summary)