def get_row_data(row, column_name, cat_time_ns = True):
    """
    Retrieves the requested column's data from the given row.
    
    @cat_time_ns: If the column_name has "_time" in it, will concatenate 
    the column with any column having the same name but "_time_ns".
    """
    column_name_ns = re.sub(r'_time', r'_time_ns', column_name)
    try:
        rowattrs = [attr for attr in row.__slots__]
    except AttributeError:
        rowattrs = [attr for attr in row.__dict__.iterkeys()]

    if cat_time_ns and "_time" in column_name and column_name_ns in rowattrs:
        return int(getattr(row, column_name)) + 10**(-9.)*int(getattr(row, column_name_ns))
    else:
        return getattr(row, column_name)