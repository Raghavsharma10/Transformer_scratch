def load_polyphen_annotations(self, as_dataframe=False,
                                  filter_fn=None):
        """Load a dataframe containing polyphen2 annotations for all variants

        Parameters
        ----------
        database_file : string, sqlite
            Path to the WHESS/Polyphen2 SQLite database.
            Can be downloaded and bunzip2"ed from http://bit.ly/208mlIU
        filter_fn : function
            Takes a FilterablePolyphen and returns a boolean.
            Only annotations returning True are preserved.
            Overrides default self.filter_fn. `None` passes through to self.filter_fn.

        Returns
        -------
        annotations
            Dictionary of patient_id to a DataFrame that contains annotations
        """
        filter_fn = first_not_none_param([filter_fn, self.filter_fn], no_filter)
        patient_annotations = {}
        for patient in self:
            annotations = self._load_single_patient_polyphen(
                patient,
                filter_fn=filter_fn)
            if annotations is not None:
                annotations["patient_id"] = patient.id
                patient_annotations[patient.id] = annotations
        if as_dataframe:
            return pd.concat(patient_annotations.values())
        return patient_annotations