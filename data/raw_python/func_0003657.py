def dict_list_2_matrix(dict_list, columns):
    """
        >>> dict_list_2_matrix([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], ('a', 'b'))
        [[1, 2], [3, 4]]

    :param dict_list: 字典列表
    :param columns: 字典的键
    """
    k = len(columns)
    n = len(dict_list)

    result = [[None] * k for i in range(n)]
    for i in range(n):
        row = dict_list[i]
        for j in range(k):
            col = columns[j]
            if col in row:
                result[i][j] = row[col]
            else:
                result[i][j] = None
    return result