def write_list_to_csv(list_of_rows, filepath, headers=None):
    # type: (List[Union[DictUpperBound, List]], str, Union[int, List[int], List[str], None]) -> None
    """Write a list of rows in dict or list form to a csv. (The headers argument is either a row
       number or list of row numbers (in case of multi-line headers) to be considered as headers
       (rows start counting at 1), or the actual headers defined a list of strings. If not set,
       all rows will be treated as containing values.)

    Args:
        list_of_rows (List[Union[DictUpperBound, List]]): List of rows in dict or list form
        filepath (str): Path to write to
        headers (Union[int, List[int], List[str], None]): Headers to write. Defaults to None.

    Returns:
        None

    """
    stream = Stream(list_of_rows, headers=headers)
    stream.open()
    stream.save(filepath, format='csv')
    stream.close()