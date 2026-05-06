def list_classification_predictors(self):
        """List available classification predictors."""
        preds = [self.create(x) for x in self._predictors.keys()]
        return [x.name for x in preds if x.ptype == "classification"]