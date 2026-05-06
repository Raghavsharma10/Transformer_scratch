def create_decision_tree(data, attributes, class_attr, fitness_func, wrapper, **kwargs):
    """
    Returns a new decision tree based on the examples given.
    """
    
    split_attr = kwargs.get('split_attr', None)
    split_val = kwargs.get('split_val', None)
    
    assert class_attr not in attributes
    node = None
    data = list(data) if isinstance(data, Data) else data
    if wrapper.is_continuous_class:
        stop_value = CDist(seq=[r[class_attr] for r in data])
        # For a continuous class case, stop if all the remaining records have
        # a variance below the given threshold.
        stop = wrapper.leaf_threshold is not None \
            and stop_value.variance <= wrapper.leaf_threshold
    else:
        stop_value = DDist(seq=[r[class_attr] for r in data])
        # For a discrete class, stop if all remaining records have the same
        # classification.
        stop = len(stop_value.counts) <= 1

    if not data or len(attributes) <= 0:
        # If the dataset is empty or the attributes list is empty, return the
        # default value. The target attribute is not in the attributes list, so
        # we need not subtract 1 to account for the target attribute.
        if wrapper:
            wrapper.leaf_count += 1
        return stop_value
    elif stop:
        # If all the records in the dataset have the same classification,
        # return that classification.
        if wrapper:
            wrapper.leaf_count += 1
        return stop_value
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(
            data,
            attributes,
            class_attr,
            fitness_func,
            method=wrapper.metric)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
#        tree = {best:{}}
        node = Node(tree=wrapper, attr_name=best)
        node.n += len(data)

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                [r for r in data if r[best] == val],
                [attr for attr in attributes if attr != best],
                class_attr,
                fitness_func,
                split_attr=best,
                split_val=val,
                wrapper=wrapper)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            if isinstance(subtree, Node):
                node._branches[val] = subtree
            elif isinstance(subtree, (CDist, DDist)):
                node.set_leaf_dist(attr_value=val, dist=subtree)
            else:
                raise Exception("Unknown subtree type: %s" % (type(subtree),))

    return node