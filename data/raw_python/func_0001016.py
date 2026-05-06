def _get_predictions(self):
        """
        get predictions using internal least squares/supplied regressor
        """
        if self.regressor is None:
            preds = safe_sparse_dot(self.hidden_activations_, self.coefs_)
        else:
            preds = self.regressor.predict(self.hidden_activations_)

        return preds