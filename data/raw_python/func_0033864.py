def julianDay(year, month, day):
    "returns julian day=day since Jan 1 of year"
    hr = 12  #make sure you fall into right day, middle is save
    t = time.mktime((year, month, day, hr, 0, 0.0, 0, 0, -1))
    julDay = time.localtime(t)[7]
    return julDay