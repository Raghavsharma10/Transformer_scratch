def quote_sql(value, param=None):
    """
    USED TO EXPAND THE PARAMETERS TO THE SQL() OBJECT
    """
    try:
        if isinstance(value, SQL):
            if not param:
                return value
            param = {k: quote_sql(v) for k, v in param.items()}
            return SQL(expand_template(value, param))
        elif is_text(value):
            return SQL(value)
        elif is_data(value):
            return quote_value(json_encode(value))
        elif hasattr(value, '__iter__'):
            return quote_list(value)
        else:
            return text_type(value)
    except Exception as e:
        Log.error("problem quoting SQL", e)