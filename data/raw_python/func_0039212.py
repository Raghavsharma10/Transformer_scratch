def timetopythonvalue(time_val):
    "Convert a time or time range from ArcGIS REST server format to Python"
    if isinstance(time_val, sequence):
        return map(timetopythonvalue, time_val)
    elif isinstance(time_val, numeric):
        return datetime.datetime(*(time.gmtime(time_val))[:6])
    elif isinstance(time_val, numeric):
        values = []
        try:
            values = map(long, time_val.split(","))
        except:
            pass
        if values:
            return map(timetopythonvalue, values)
    raise ValueError(repr(time_val))