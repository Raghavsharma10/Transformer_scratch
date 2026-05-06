def predict(self, fitted):
        """Assign the most likely modality given the fitted data

        Parameters
        ----------
        fitted : pandas.DataFrame or pandas.Series
            Either a (n_modalities, features) DatFrame or (n_modalities,)
            Series, either of which will return the best modality for each
            feature.
        """
        if fitted.shape[0] != len(self.modalities):
            raise ValueError("This data doesn't look like it had the distance "
                             "between it and the five modalities calculated")
        return fitted.idxmin()