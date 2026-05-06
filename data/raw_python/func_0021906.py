def load_variants(self, patients=None, filter_fn=None, **kwargs):
        """Load a dictionary of patient_id to varcode.VariantCollection

        Parameters
        ----------
        patients : str, optional
            Filter to a subset of patients
        filter_fn : function
            Takes a FilterableVariant and returns a boolean. Only variants returning True are preserved.
            Overrides default self.filter_fn. `None` passes through to self.filter_fn.

        Returns
        -------
        merged_variants
            Dictionary of patient_id to VariantCollection
        """
        filter_fn = first_not_none_param([filter_fn, self.filter_fn], no_filter)
        filter_fn_name = self._get_function_name(filter_fn)
        logger.debug("loading variants with filter_fn: {}".format(filter_fn_name))
        patient_variants = {}

        for patient in self.iter_patients(patients):
            variants = self._load_single_patient_variants(patient, filter_fn, **kwargs)
            if variants is not None:
                patient_variants[patient.id] = variants
        return patient_variants