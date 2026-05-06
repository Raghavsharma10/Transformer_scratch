def output_tree_ensemble(tree_ensemble_obj, output_filename, attribute_names=None):
    """
    Write each decision tree in an ensemble to a file.

    Parameters
    ----------
    tree_ensemble_obj : sklearn.ensemble object
        Random Forest or Gradient Boosted Regression object
    output_filename : str
        File where trees are written
    attribute_names : list
        List of attribute names to be used in place of indices if available.
    """
    for t, tree in enumerate(tree_ensemble_obj.estimators_):
        print("Writing Tree {0:d}".format(t))
        out_file = open(output_filename + ".{0:d}.tree", "w")
        #out_file.write("Tree {0:d}\n".format(t))
        tree_str = print_tree_recursive(tree.tree_, 0, attribute_names)
        out_file.write(tree_str)
        #out_file.write("\n")
        out_file.close()
    return