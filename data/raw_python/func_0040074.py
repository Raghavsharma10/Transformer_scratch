def excel_to_sql(excel_file_path, engine,
                 read_excel_kwargs=None,
                 to_generic_type_kwargs=None,
                 to_sql_kwargs=None):
    """Create a database from excel.

    :param read_excel_kwargs: dict, arguments for ``pandas.read_excel`` method.
      example: ``{"employee": {"skiprows": 10}, "department": {}}``
    :param to_sql_kwargs: dict, arguments for ``pandas.DataFrame.to_sql`` 
      method.

    limitation:

    1. If a integer column has None value, data type in database will be float.
      Because pandas thinks that it is ``np.nan``.
    2. If a string column looks like integer, ``pandas.read_excel()`` method
      doesn't have options to convert it to string.
    """
    if read_excel_kwargs is None:
        read_excel_kwargs = dict()

    if to_sql_kwargs is None:
        to_sql_kwargs = dict()

    if to_generic_type_kwargs is None:
        to_generic_type_kwargs = dict()

    xl = pd.ExcelFile(excel_file_path)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(
            excel_file_path, sheet_name,
            **read_excel_kwargs.get(sheet_name, dict())
        )

        kwargs = to_generic_type_kwargs.get(sheet_name)
        if kwargs:
            data = to_dict_list_generic_type(df, **kwargs)
            smart_insert(data, sheet_name, engine)
        else:
            df.to_sql(
                sheet_name, engine, index=False,
                **to_sql_kwargs.get(sheet_name, dict(if_exists="replace"))
            )