def quote_value(value):
    """
    convert values to mysql code for the same
    mostly delegate directly to the mysql lib, but some exceptions exist
    """
    try:
        if value == None:
            return SQL_NULL
        elif isinstance(value, SQL):
            return quote_sql(value.template, value.param)
        elif is_text(value):
            return SQL("'" + "".join(ESCAPE_DCT.get(c, c) for c in value) + "'")
        elif is_data(value):
            return quote_value(json_encode(value))
        elif is_number(value):
            return SQL(text_type(value))
        elif isinstance(value, datetime):
            return SQL("str_to_date('" + value.strftime("%Y%m%d%H%M%S.%f") + "', '%Y%m%d%H%i%s.%f')")
        elif isinstance(value, Date):
            return SQL("str_to_date('" + value.format("%Y%m%d%H%M%S.%f") + "', '%Y%m%d%H%i%s.%f')")
        elif hasattr(value, '__iter__'):
            return quote_value(json_encode(value))
        else:
            return quote_value(text_type(value))
    except Exception as e:
        Log.error("problem quoting SQL {{value}}", value=repr(value), cause=e)