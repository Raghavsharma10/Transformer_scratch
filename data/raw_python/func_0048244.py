def hourly(place):
    """return data as list of dicts with all data filled in."""
    # time in utc?
    lat, lon = place
    url = "https://api.forecast.io/forecast/%s/%s,%s?solar" % (APIKEY, lat,
                                                               lon)
    w_data = json.loads(urllib2.urlopen(url).read())
    hourly_data = w_data['hourly']['data']
    mangled = []
    for i in hourly_data:
        mangled.append(mangle(i))
    return mangled