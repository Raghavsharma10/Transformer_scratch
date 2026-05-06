def group_keys(shape, *inputs_keys):
    """
    Usecase: Two sets of chunks, one spans the whole of a dimension, the other
    chunked it up.  We need to know that we need to collect together the
    chunked form, so that we can work with both sets at the same time.

    Conceptually we have multiple source inputs, each with multiple key sets
    for indexing.

    NOTE: We treat the grouping independently per dimension. In practice this
    means we may be grouping more than is strictly necessary if we were being
    smart about multi-dimensional grouping. Anecdotally, that optimisation is
    currently not worth the implementation effort.

    """
    # Store the result as a slice mapping to a subset of the inputs_keys. We
    # start with the assumption that there will be only one group, and
    # subdivide when we find this not to be the case.
    ndim = len(inputs_keys[0][0])
    grouped_inputs_keys = {tuple((None, None, None)
                                 for _ in range(ndim)): inputs_keys}

    for dim, dim_len in enumerate(shape):
        # Compute the groups for this dimension.
        for group_keys, group_inputs_keys in grouped_inputs_keys.copy(
                                                                 ).items():
            group_inputs_key_for_dim = [[keys[dim] for keys in input_keys]
                                        for input_keys in group_inputs_keys]
            grouped_inputs_key = dimension_group_to_lowest_common(
                    dim_len, group_inputs_key_for_dim).items()
            # If this group hasn't sub-divided, continue on to next group.
            if len(grouped_inputs_key) == 1:
                continue
            else:
                # Drop the bigger group from the result dictionary and in its
                # place, add all of the subgroups.
                grouped_inputs_keys.pop(group_keys)
                # Make the group keys mutable so that we can inject our
                # subgroups.
                group_keys = list(group_keys)
                group_inputs_keys = list(group_inputs_keys)
                for subgroup_key, subgroup_inputs_key in grouped_inputs_key:
                    group_keys[dim] = subgroup_key

                    # Start with an empty list, one for each input.
                    subgroup_inputs_keys = [[] for _ in subgroup_inputs_key]
                    per_input = zip(group_inputs_keys, subgroup_inputs_key,
                                    subgroup_inputs_keys)
                    for (input_keys, subgroup_input_key,
                         new_input_keys) in per_input:
                        for keys in input_keys[:]:
                            norm_key = normalize_slice(keys[dim], dim_len)
                            if norm_key in subgroup_input_key:
                                input_keys.remove(keys)
                                new_input_keys.append(keys)
                    subgroup_inputs_keys = tuple(subgroup_inputs_keys)

                    grouped_inputs_keys[tuple(
                        group_keys)] = subgroup_inputs_keys
    return grouped_inputs_keys