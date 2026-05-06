def _load_single_patient_cufflinks(self, patient, filter_ok):
        """
        Load Cufflinks gene quantification given a patient

        Parameters
        ----------
        patient : Patient
        filter_ok : bool, optional
            If true, filter Cufflinks data to row with FPKM_status == "OK"

        Returns
        -------
        data: Pandas dataframe
            Pandas dataframe of sample's Cufflinks data
            columns include patient_id, gene_id, gene_short_name, FPKM, FPKM_conf_lo, FPKM_conf_hi
        """
        data = pd.read_csv(patient.tumor_sample.cufflinks_path, sep="\t")
        data["patient_id"] = patient.id

        if filter_ok:
            # Filter to OK FPKM counts
            data = data[data["FPKM_status"] == "OK"]
        return data