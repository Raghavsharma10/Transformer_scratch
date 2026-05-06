def _get_best_prediction(self, record, train=True):
        """
        Gets the prediction from the tree with the lowest mean absolute error.
        """
        if not self.trees:
            return
        best = (+1e999999, None)
        for tree in self.trees:
            best = min(best, (tree.mae.mean, tree))
        _, best_tree = best
        prediction, tree_mae = best_tree.predict(record, train=train)
        return prediction.mean