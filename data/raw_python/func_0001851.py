def read_list_from_csv(filepath, dict_form=False, headers=None, **kwargs):
    # type: (str, bool, Union[int, List[int], List[str], None], Any) -> List[Union[Dict, List]]
    """Read a list of rows in dict or list form from a csv. (The headers argument is either a row
       number or list of row numbers (in case of multi-line headers) to be considered as headers
       (rows start counting at 1), or the actual headers defined a list of strings. If not set,
       all rows will be treated as containing values.)

    Args:
        filepath (str): Path to read from
        dict_form (bool): Return in dict form. Defaults to False.
        headers (Union[int, List[int], List[str], None]): Row number of headers. Defaults to None.
        **kwargs: Other arguments to pass to Tabulator Stream

    Returns:
        List[Union[Dict, List]]: List of rows in dict or list form

    """
    stream = Stream(filepath, headers=headers, **kwargs)
    stream.open()
    result = stream.read(keyed=dict_form)
    stream.close()
    return result