def _load_single_patient_kallisto(self, patient):
        """
        Load Kallisto gene quantification given a patient

        Parameters
        ----------
        patient : Patient

        Returns
        -------
        data: Pandas dataframe
            Pandas dataframe of sample's Kallisto data
            columns include patient_id, target_id, length, eff_length, est_counts, tpm
        """
        data = pd.read_csv(patient.tumor_sample.kallisto_path, sep="\t")
        data["patient_id"] = patient.id
        return data