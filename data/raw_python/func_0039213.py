def pythonvaluetotime(time_val):
    "Convert a time or time range from Python datetime to ArcGIS REST server"
    if time_val is None:
        return None
    elif isinstance(time_val, numeric):
        return str(long(time_val * 1000.0))
    elif isinstance(time_val, date):
        dtlist = [time_val.year, time_val.month, time_val.day]
        if isinstance(time_val, datetime.datetime):
            dtlist += [time_val.hour, time_val.minute, time_val.second]
        else:
            dtlist += [0, 0, 0]
        return long(calendar.timegm(dtlist) * 1000.0)
    elif (isinstance(time_val, sequence)
                    and len(time_val) == 2):
        if all(isinstance(x, numeric) 
               for x in time_val):
            return ",".join(pythonvaluetotime(x) 
                            for x in time_val)
        elif all(isinstance(x, date) 
               for x in time_val):
            return ",".join(pythonvaluetotime(x) 
                            for x in time_val)
    raise ValueError(repr(time_val))