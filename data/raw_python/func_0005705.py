def check_that_operator_can_be_applied_to_produces_items(op, g1, g2):
    """
    Helper function to check that the operator `op` can be applied to items produced by g1 and g2.
    """
    g1_tmp_copy = g1.spawn()
    g2_tmp_copy = g2.spawn()
    sample_item_1 = next(g1_tmp_copy)
    sample_item_2 = next(g2_tmp_copy)
    try:
        op(sample_item_1, sample_item_2)
    except TypeError:
        raise TypeError(f"Operator '{op.__name__}' cannot be applied to items produced by {g1} and {g2} "
                        f"(which have type {type(sample_item_1)} and {type(sample_item_2)}, respectively)")