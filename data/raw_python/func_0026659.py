def compare_dicts(old_full, new_full, old_data, new_data, depth=0):
    """Function compares dictionaries by key-value recursively.

    Old and new input data are both dictionaries
    """
    depth = depth + 1
    indent = "  "*depth

    # Print with an indentation matching the nested-dictionary depth
    def my_print(str):
        print("{}{}".format(indent, str))

    old_keys = list(old_data.keys())
    # Compare data key by key, in *this* dictionary level
    # Note: since we're comparing by keys explicity, order doesnt matter
    for key in old_keys:
        # Remove elements as we go
        old_vals = old_data.pop(key)
        # Current key
        my_print("{}".format(key))
        # If `new_data` doesnt also have this key, return False
        if key not in new_data:
            my_print("Key '{}' not in new_data.".format(key))
            my_print("Old:")
            my_print(pprint(new_data))
            my_print("New:")
            my_print(pprint(new_data))
            return False

        # If it does have the key, extract the values (remove as we go)
        new_vals = new_data.pop(key)
        # If these values are a sub-dictionary, compare those
        if isinstance(old_vals, dict) and isinstance(new_vals, dict):
            # If the sub-dictionary are not the same, return False
            if not compare_dicts(old_full, new_full, old_vals, new_vals, depth=depth):
                return False
        # If these values are a list of sub-dictionaries, compare each of those
        elif (isinstance(old_vals, list) and isinstance(old_vals[0], dict) and
              isinstance(old_vals, list) and isinstance(old_vals[0], dict)):
            for old_elem, new_elem in zip_longest(old_vals, new_vals):
                # If one or the other has extra elements, print message, but
                # continue on
                if old_elem is None or new_elem is None:
                    my_print("Missing element!")
                    my_print("\tOld: '{}'".format(old_elem))
                    my_print("\tNew: '{}'".format(new_elem))
                else:
                    if not compare_dicts(old_full, new_full, old_elem, new_elem, depth=depth):
                        return False

        # At the lowest-dictionary level, compare the values themselves
        else:
            # Turn everything into a list for convenience (most things should be
            # already)
            if  (not isinstance(old_vals, list) and
                 not isinstance(new_vals, list)):
                old_vals = [old_vals]
                new_vals = [new_vals]

            # Sort both lists
            old_vals = sorted(old_vals)
            new_vals = sorted(new_vals)

            for oldv, newv in zip_longest(old_vals, new_vals):
                # If one or the other has extra elements, print message, but
                # continue on
                if oldv is None or newv is None:
                    my_print("Missing element!")
                    my_print("\tOld: '{}'".format(oldv))
                    my_print("\tNew: '{}'".format(newv))
                # If values match, continue
                elif oldv == newv:
                    my_print("Good Match: '{}'".format(key))
                # If values dont match, return False
                else:
                    my_print("Bad  Match: '{}'".format(key))
                    my_print("\tOld: '{}'".format(oldv))
                    my_print("\tNew: '{}'".format(newv))
                    return False

    return True