def herp_derp_interp(place):
    """simple interpolation of GFS forecast"""
    lat, lon = place
    #begin=2014-02-14T00%3A00%3A00&end=2018-02-22T00%3A00%3A00
    fmt = '%Y-%m-%dT00:00:00'
    fmt = '%Y-%m-%dT%H:%M:00'
    begin = (datetime.datetime.now()-datetime.timedelta(hours=12)).strftime(fmt)
    #end=(datetime.datetime.now()+datetime.timedelta(hours=48)).strftime(fmt)
    url = "http://graphical.weather.gov/xml/SOAP_server/ndfdXMLclient.php?" + \
            "whichClient=NDFDgen&lat=%s&lon=%s&" % (lat, lon) + \
            "Unit=e&temp=temp&wspd=wspd&sky=sky&wx=wx&rh=rh&" + \
            "product=time-series&begin=%s&end=2018-02-22T00:00:00" % begin + \
            "&Submit=Submit"""
    res = urllib2.urlopen(url).read()
    root = ET.fromstring(res)

    time_series = [_cast_float(i.text) for i in \
            root.findall('./data/time-layout')[0].iterfind('start-valid-time')]
    #knots to mph
    wind_speed = [eval(i.text)*1.15 for i in \
            root.findall('./data/parameters/wind-speed')[0].iterfind('value')]
    cloud_cover = [eval(i.text)/100.0 for i in \
            root.findall('./data/parameters/cloud-amount')[0].iterfind('value')]
    temperature = [eval(i.text) for i in \
            root.findall('./data/parameters/temperature')[0].iterfind('value')]

    ws_interp = interp1d(time_series, wind_speed, kind='cubic')
    cc_interp = interp1d(time_series, cloud_cover, kind='cubic')
    t_interp = interp1d(time_series, temperature, kind='cubic')
    start_date = datetime.datetime.utcfromtimestamp(time_series[0])

    series = []
    for i in range(48):
        try:
            temp_dict = {}
            forecast_dt = start_date + datetime.timedelta(hours=i)
            temp_dict['utc_datetime'] = forecast_dt
            temp_dict['windSpeed'] = ws_interp(_cast_float(forecast_dt)).item()
            temp_dict['temperature'] = t_interp(_cast_float(forecast_dt)).item()
            temp_dict['cloudCover'] = cc_interp(_cast_float(forecast_dt)).item()
            series.append(temp_dict)
        except:
            pass
    return series