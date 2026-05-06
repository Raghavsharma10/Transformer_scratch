def load_kallisto(self):
        """
        Load Kallisto transcript quantification data for a cohort

        Parameters
        ----------

        Returns
        -------
        kallisto_data : Pandas dataframe
            Pandas dataframe with Kallisto data for all patients
            columns include patient_id, gene_name, est_counts
        """
        kallisto_data = pd.concat(
            [self._load_single_patient_kallisto(patient) for patient in self],
            copy=False
        )

        if self.kallisto_ensembl_version is None:
            raise ValueError("Required a kallisto_ensembl_version but none was specified")

        ensembl_release = cached_release(self.kallisto_ensembl_version)

        kallisto_data["gene_name"] = \
            kallisto_data["target_id"].map(lambda t: ensembl_release.gene_name_of_transcript_id(t))

        # sum counts across genes
        kallisto_data = \
            kallisto_data.groupby(["patient_id", "gene_name"])[["est_counts"]].sum().reset_index()

        return kallisto_data