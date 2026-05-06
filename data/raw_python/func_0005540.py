def ufloatDict_stdev(self, ufloat_dict):
        'This gives us a dictionary of nominal values from a dictionary of uncertainties'
        return OrderedDict(izip(ufloat_dict.keys(), map(lambda x: x.std_dev, ufloat_dict.values())))