def current(place):
    """return data as list of dicts with all data filled in."""
    lat, lon = place
    url = "https://api.forecast.io/forecast/%s/%s,%s?solar" % (APIKEY, lat,
                                                               lon)
    w_data = json.loads(urllib2.urlopen(url).read())
    currently = w_data['currently']
    return mangle(currently)