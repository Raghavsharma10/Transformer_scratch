def categorize(
    data, col_name: str = None, new_col_name: str = None,
    categories: dict = None, max_categories: float = 0.15
):
    """
    :param data:
    :param col_name:
    :param new_col_name:
    :param categories:
    :param max_categories: max proportion threshold of categories
    :return: new categories
    :rtype dict:
    """

    _categories = {}
    if col_name is None:
        if categories is not None:
            raise Exception(
                'col_name is None when categories was defined.'
            )
        # create a list of cols with all object columns
        cols = [
            k for k in data.keys()
            if data[k].dtype == 'object' and
            (data[k].unique() / data[k].count()) <= max_categories
        ]
    else:
        # create a list with col_name
        if new_col_name is not None:
            data[new_col_name] = data[col_name]
            col_name = new_col_name

        cols = [col_name]

    for c in cols:
        if categories is not None:
            # assert all keys is a number
            assert all(type(k) in (int, float) for k in categories.keys())
            # replace values using given categories dict
            data[c].replace(categories, inplace=True)
            # change column to categorical type
            data[c] = data[c].astype('category')
            # update categories information
            _categories.update({c: categories})
        else:
            # change column to categorical type
            data[c] = data[c].astype('category')
            # change column to categorical type
            _categories.update({
                c: dict(enumerate(
                    data[c].cat.categories,
                ))
            })
    return _categories