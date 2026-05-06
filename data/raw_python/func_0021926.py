def summarize_data_sources(self):
        """Utility function to summarize data source status for this Cohort, useful for confirming
        the state of data used for an analysis

        Returns
        ----------
        Dictionary with summary of data sources

        Currently contains
        - dataframe_hash: hash of the dataframe (see `?cohorts.Cohort.summarize_dataframe`)
        - provenance_file_summary: summary of provenance file contents (see `?cohorts.Cohort.summarize_provenance`)
        """
        provenance_file_summary = self.summarize_provenance()
        dataframe_hash = self.summarize_dataframe()
        results = {
            "provenance_file_summary": provenance_file_summary,
            "dataframe_hash": dataframe_hash
        }
        return(results)