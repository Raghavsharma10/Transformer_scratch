def epochdate(timestamp):
    '''
    Convet an epoch date to a tuple in format ("yyyy-mm-dd","hh:mm:ss")
    Example: "1023456427" -> ("2002-06-07","15:27:07")

    Parameters:
    - `timestamp`: date in epoch format
    '''

    dt = datetime.fromtimestamp(float(timestamp)).timetuple()
    fecha = "{0:d}-{1:02d}-{2:02d}".format(dt.tm_year, dt.tm_mon, dt.tm_mday)
    hora = "{0:02d}:{1:02d}:{2:02d}".format(dt.tm_hour, dt.tm_min, dt.tm_sec)
    return (fecha, hora)