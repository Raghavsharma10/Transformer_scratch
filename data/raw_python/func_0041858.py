def loadjson(filename, asnparrays=False):
    """Load data from text file in JSON format.

    Numpy arrays are converted if specified with the `asnparrays` keyword
    argument. Note that this only works to the second level of the dictionary.
    Returns a single dict.
    """
    with open(filename) as f:
        data = json.load(f)
    if asnparrays:
        for key, value in data.items():
            if type(value) is list:
                data[key] = np.asarray(value)
            if type(value) is dict:
                for key2, value2 in value.items():
                    if type(value2) is list:
                        data[key][key2] = np.asarray(value2)
    return data