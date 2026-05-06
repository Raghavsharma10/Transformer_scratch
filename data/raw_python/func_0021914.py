def load_cufflinks(self, filter_ok=True):
        """
        Load a Cufflinks gene expression data for a cohort

        Parameters
        ----------
        filter_ok : bool, optional
            If true, filter Cufflinks data to row with FPKM_status == "OK"

        Returns
        -------
        cufflinks_data : Pandas dataframe
            Pandas dataframe with Cufflinks data for all patients
            columns include patient_id, gene_id, gene_short_name, FPKM, FPKM_conf_lo, FPKM_conf_hi
        """
        return \
            pd.concat(
                [self._load_single_patient_cufflinks(patient, filter_ok) for patient in self],
                copy=False
        )