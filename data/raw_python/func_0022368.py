def epomeo_gpx(data_set='epomeo_gpx', sample_every=4):
    """Data set of three GPS traces of the same movement on Mt Epomeo in Ischia. Requires gpxpy to run."""
    import gpxpy
    import gpxpy.gpx
    if not data_available(data_set):
        download_data(data_set)
    files = ['endomondo_1', 'endomondo_2', 'garmin_watch_via_endomondo','viewranger_phone', 'viewranger_tablet']

    X = []
    for file in files:
        gpx_file = open(os.path.join(data_path, 'epomeo_gpx', file + '.gpx'), 'r')

        gpx = gpxpy.parse(gpx_file)
        segment = gpx.tracks[0].segments[0]
        points = [point for track in gpx.tracks for segment in track.segments for point in segment.points]
        data = [[(point.time-datetime.datetime(2013,8,21)).total_seconds(), point.latitude, point.longitude, point.elevation] for point in points]
        X.append(np.asarray(data)[::sample_every, :])
        gpx_file.close()
    if pandas_available:
        X = pd.DataFrame(X[0], columns=['seconds', 'latitude', 'longitude', 'elevation'])
        X.set_index(keys='seconds', inplace=True)
    return data_details_return({'X' : X, 'info' : 'Data is an array containing time in seconds, latitude, longitude and elevation in that order.'}, data_set)