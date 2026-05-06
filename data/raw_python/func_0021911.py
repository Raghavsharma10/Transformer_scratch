def load_effects(self, patients=None, only_nonsynonymous=False,
                     all_effects=False, filter_fn=None, **kwargs):
        """Load a dictionary of patient_id to varcode.EffectCollection

        Note that this only loads one effect per variant.

        Parameters
        ----------
        patients : str, optional
            Filter to a subset of patients
        only_nonsynonymous : bool, optional
            If true, load only nonsynonymous effects, default False
        all_effects : bool, optional
            If true, return all effects rather than only the top-priority effect per variant
        filter_fn : function
            Takes a FilterableEffect and returns a boolean. Only effects returning True are preserved.
            Overrides default self.filter_fn. `None` passes through to self.filter_fn.

        Returns
        -------
        effects
             Dictionary of patient_id to varcode.EffectCollection
        """
        filter_fn = first_not_none_param([filter_fn, self.filter_fn], no_filter)
        filter_fn_name = self._get_function_name(filter_fn)
        logger.debug("loading effects with filter_fn {}".format(filter_fn_name))
        patient_effects = {}
        for patient in self.iter_patients(patients):
            effects = self._load_single_patient_effects(
                patient, only_nonsynonymous, all_effects, filter_fn, **kwargs)
            if effects is not None:
                patient_effects[patient.id] = effects
        return patient_effects