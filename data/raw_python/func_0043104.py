def leaves(value, prefix=None):
    """
    LIKE items() BUT RECURSIVE, AND ONLY FOR THE LEAVES (non dict) VALUES
    SEE wrap_leaves FOR THE INVERSE

    :param value: THE Mapping TO TRAVERSE
    :param prefix:  OPTIONAL PREFIX GIVEN TO EACH KEY
    :return: Data, WHICH EACH KEY BEING A PATH INTO value TREE
    """
    prefix = coalesce(prefix, "")
    output = []
    for k, v in value.items():
        try:
            if _get(v, CLASS) in data_types:
                output.extend(leaves(v, prefix=prefix + literal_field(k) + "."))
            else:
                output.append((prefix + literal_field(k), unwrap(v)))
        except Exception as e:
            get_logger().error("Do not know how to handle", cause=e)
    return output