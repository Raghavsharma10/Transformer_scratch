def _make_graphite_api_points_list(influxdb_data):
    """Make graphite-api data points dictionary from Influxdb ResultSet data"""
    _data = {}
    for key in influxdb_data.keys():
        _data[key[0]] = [(datetime.datetime.fromtimestamp(float(d['time'])),
                          d['value']) for d in influxdb_data.get_points(key[0])]
    return _data