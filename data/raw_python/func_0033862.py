def dayOfWeek(year, month, day):
    "returns day of week: 0=Sun, 1=Mon, .., 6=Sat"
    hr = 12  #make sure you fall into right day, middle is save
    t = time.mktime((year, month, day, hr, 0, 0.0, 0, 0, -1))
    pyDow = time.localtime(t)[6]
    gpsDow = (pyDow + 1) % 7
    return gpsDow