def ready_to_split(self):
        """
        Returns true if this node is ready to branch off additional nodes.
        Returns false otherwise.
        """
        # Never split if we're a leaf that predicts adequately.
        threshold = self._tree.leaf_threshold
        if self._tree.data.is_continuous_class:
            var = self._class_cdist.variance
            if var is not None and threshold is not None \
            and var <= threshold:
                return False
        else:
            best_prob = self._class_ddist.best_prob
            if best_prob is not None and threshold is not None \
            and best_prob >= threshold:
                return False
            
        return self._tree.auto_grow \
            and not self.attr_name \
            and self.n >= self._tree.splitting_n