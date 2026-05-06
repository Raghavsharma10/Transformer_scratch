def flatten_list(multiply_list):
    """
    碾平 list::

        >>> a = [1, 2, [3, 4], [[5, 6], [7, 8]]]
        >>> flatten_list(a)
        [1, 2, 3, 4, 5, 6, 7, 8]

    :param multiply_list: 混淆的多层列表
    :return: 单层的 list
    """
    if isinstance(multiply_list, list):
        return [rv for l in multiply_list for rv in flatten_list(l)]
    else:
        return [multiply_list]