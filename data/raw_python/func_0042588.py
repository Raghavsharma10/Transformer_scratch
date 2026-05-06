def getYamlDocument(filePath):
    """
    Return a yaml file's contents as a dictionary
    """
    with open(filePath) as stream:
        doc = yaml.load(stream)
        return doc