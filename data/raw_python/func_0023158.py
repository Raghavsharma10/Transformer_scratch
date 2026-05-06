def _many_to_one(input_dict):
    """Convert a many-to-one mapping to a one-to-one mapping"""
    return dict((key, val)
                for keys, val in input_dict.items()
                for key in keys)