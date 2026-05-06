def modify_filename_id(filename):
    """Modify filename to have a unique numerical identifier."""
    split_filename = os.path.splitext(filename)
    id_num_re = re.compile('(\(\d\))')
    id_num = re.findall(id_num_re, split_filename[-2])
    if id_num:
        new_id_num = int(id_num[-1].lstrip('(').rstrip(')')) + 1

        # Reconstruct filename with incremented id and its extension
        filename = ''.join((re.sub(id_num_re, '({0})'.format(new_id_num),
                                   split_filename[-2]), split_filename[-1]))
    else:
        split_filename = os.path.splitext(filename)

        # Reconstruct filename with new id and its extension
        filename = ''.join(('{0} (2)'.format(split_filename[-2]),
                            split_filename[-1]))
    return filename