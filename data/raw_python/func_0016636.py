def mean_oob_mae_weight(trees):
        """
        Returns weights proportional to the out-of-bag mean absolute error for each tree.
        """
        weights = []
        active_trees = []
        for tree in trees:
            oob_mae = tree.out_of_bag_mae
            if oob_mae is None or oob_mae.mean is None:
                continue
            weights.append(oob_mae.mean)
            active_trees.append(tree)
        if not active_trees:
            return
        weights = normalize(weights)
        return zip(weights, active_trees)