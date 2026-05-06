def get_unique_groups(input_list):
    """Function to get a unique list of groups."""
    out_list = []
    for item in input_list:
        if item not in out_list:
            out_list.append(item)
    return out_list