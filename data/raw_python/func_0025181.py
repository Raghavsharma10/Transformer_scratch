def read_geojson(filename):
    """
    Reads a geojson file containing an STObject and initializes a new STObject from the information in the file.

    Args:
        filename: Name of the geojson file

    Returns:
        an STObject
    """
    json_file = open(filename)
    data = json.load(json_file)
    json_file.close()
    times = data["properties"]["times"]
    main_data = dict(timesteps=[], masks=[], x=[], y=[], i=[], j=[])
    attribute_data = dict()
    for feature in data["features"]:
        for main_name in main_data.keys():
            main_data[main_name].append(np.array(feature["properties"][main_name]))
        for k, v in feature["properties"]["attributes"].items():
            if k not in attribute_data.keys():
                attribute_data[k] = [np.array(v)]
            else:
                attribute_data[k].append(np.array(v))
    kwargs = {}
    for kw in ["dx", "step", "u", "v"]:
        if kw in data["properties"].keys():
            kwargs[kw] = data["properties"][kw]
    sto = STObject(main_data["timesteps"], main_data["masks"], main_data["x"], main_data["y"],
                   main_data["i"], main_data["j"], times[0], times[-1], **kwargs)
    for k, v in attribute_data.items():
        sto.attributes[k] = v
    return sto