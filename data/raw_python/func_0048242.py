def data(place):
    """get forecast data."""
    lat, lon = place
    url = "https://api.forecast.io/forecast/%s/%s,%s?solar" % (APIKEY, lat,
                                                               lon)
    w_data = json.loads(urllib2.urlopen(url).read())
    return w_data