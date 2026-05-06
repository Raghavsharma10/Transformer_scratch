def infer_spectra(self, ds):
        """ 
        After inferring labels for the test spectra,
        infer the model spectra and update the dataset
        model_spectra attribute.
        
        Parameters
        ----------
        ds: Dataset object
        """
        lvec_all = _get_lvec(ds.test_label_vals, self.pivots, self.scales, derivs=False)
        self.model_spectra = np.dot(lvec_all, self.coeffs.T)