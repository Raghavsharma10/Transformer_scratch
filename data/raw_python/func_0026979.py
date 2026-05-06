def dropna(data: pd.DataFrame, axis: int, **params):
    """
    Remove columns with more NA values than threshold level

    :param data:
    :param axis:
      Axes are defined for arrays with more than one dimension.
      A 2-dimensional array has two corresponding axes: the first running
      vertically downwards across rows (axis 0), and the second running
      horizontally across columns (axis 1).
      (https://docs.scipy.org/doc/numpy-1.10.0/glossary.html)
    :param params:
    :return:

    """
    if axis == 0:
        dropna_rows(data=data, **params)
    else:
        dropna_columns(data=data, **params)