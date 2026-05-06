def _get_transformation_list(key, im, fallback_sequence):
    """
    Return the list of transformations inferred from the entered key. The
    map between transform types and keys is given by module
    bogo_config (if exists) or by variable simple_telex_im

    if entered key is not in im, return "+key", meaning appending
    the entered key to current text
    """
    # if key in im:
    #     lkey = key
    # else:
    #     lkey = key.lower()
    lkey = key.lower()

    if lkey in im:
        if isinstance(im[lkey], list):
            trans_list = im[lkey]
        else:
            trans_list = [im[lkey]]

        for i, trans in enumerate(trans_list):
            if trans[0] == '<' and key.isalpha():
                trans_list[i] = trans[0] + \
                    utils.change_case(trans[1], int(key.isupper()))

        if trans_list == ['_']:
            if len(fallback_sequence) >= 2:
                # TODO Use takewhile()/dropwhile() to process the last IM keypress
                # instead of assuming it's the last key in fallback_sequence.
                t = list(map(lambda x: "_" + x,
                             _get_transformation_list(fallback_sequence[-2], im,
                                                     fallback_sequence[:-1])))
                # print(t)
                trans_list = t
            # else:
            #     trans_list = ['+' + key]

        return trans_list
    else:
        return ['+' + key]