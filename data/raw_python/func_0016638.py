def predict(self, record):
        """
        Attempts to predict the value of the class attribute by aggregating
        the predictions of each tree.
        
        Parameters:
            weighting_formula := a callable that takes a list of trees and
                returns a list of weights.
        """
        
        # Get raw predictions.
        # {tree:raw prediction}
        predictions = {}
        for tree in self.trees:
            _p = tree.predict(record)
            if _p is None:
                continue
            if isinstance(_p, CDist):
                if _p.mean is None:
                    continue
            elif isinstance(_p, DDist):
                if not _p.count:
                    continue
            predictions[tree] = _p
        if not predictions:
            return

        # Normalize weights and aggregate final prediction.
        weights = self.weighting_method(predictions.keys())
        if not weights:
            return
#        assert sum(weights) == 1.0, "Sum of weights must equal 1."
        if self.data.is_continuous_class:
            # Merge continuous class predictions.
            total = sum(w*predictions[tree].mean for w, tree in weights)
        else:
            # Merge discrete class predictions.
            total = DDist()
            for weight, tree in weights:
                prediction = predictions[tree]
                for cls_value, cls_prob in prediction.probs:
                    total.add(cls_value, cls_prob*weight)
        
        return total