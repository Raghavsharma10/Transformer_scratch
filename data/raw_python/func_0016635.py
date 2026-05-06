def best_oob_mae_weight(trees):
        """
        Returns weights so that the tree with smallest out-of-bag mean absolute error
        """
        best = (+1e999999, None)
        for tree in trees:
            oob_mae = tree.out_of_bag_mae
            if oob_mae is None or oob_mae.mean is None:
                continue
            best = min(best, (oob_mae.mean, tree))
        best_mae, best_tree = best
        if best_tree is None:
            return
        return [(1.0, best_tree)]