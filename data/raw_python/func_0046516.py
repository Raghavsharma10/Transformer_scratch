def parse_noaa_line(line):
    """Parse NOAA stations.

    This is an old list, the format is:

    NUMBER NAME & STATE/COUNTRY                     LAT   LON     ELEV (meters)

    010250 TROMSO                             NO  6941N 01855E    10
    """
    station = {}
    station['station_name'] = line[7:51].strip()
    station['station_code'] = line[0:6]
    station['CC'] = line[55:57]
    station['ELEV(m)'] = int(line[73:78])
    station['LAT'] = _mlat(line[58:64])
    station['LON'] = _mlon(line[65:71])
    station['ST'] = line[52:54]
    return station