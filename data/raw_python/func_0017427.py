def _get_features_string(self, features=None):
    """ Generates the extended newick string NHX with extra data about a node."""
    string = ""
    if features is None:
        features = []
    elif features == []:
        features = self.features

    for pr in features:
        if hasattr(self, pr):
            raw = getattr(self, pr)
            if type(raw) in ITERABLE_TYPES:
                raw = '|'.join([str(i) for i in raw])
            elif type(raw) == dict:
                raw = '|'.join(
                    map(lambda x,y: "%s-%s" %(x, y), six.iteritems(raw)))
            elif type(raw) == str:
                pass
            else:
                raw = str(raw)

            value = re.sub("["+_ILEGAL_NEWICK_CHARS+"]", "_", \
                             raw)
            if string != "":
                string +=":"
            string +="%s=%s"  %(pr, str(value))
    if string != "":
        string = "[&&NHX:"+string+"]"

    return string