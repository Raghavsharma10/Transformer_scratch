def savejson(filename, datadict):
    """Save data from a dictionary in JSON format. Note that this only
    works to the second level of the dictionary with Numpy arrays.
    """
    for key, value in datadict.items():
        if type(value) == np.ndarray:
            datadict[key] = value.tolist()
        if type(value) == dict:
            for key2, value2 in value.items():
                if type(value2) == np.ndarray:
                    datadict[key][key2] = value2.tolist()
    with open(filename, "w") as f:
        f.write(json.dumps(datadict, indent=4))