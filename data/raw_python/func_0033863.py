def gpsWeek(year, month, day):
    "returns (full) gpsWeek for given date (in UTC)"
    hr = 12  #make sure you fall into right day, middle is save
    return gpsFromUTC(year, month, day, hr, 0, 0.0)[0]