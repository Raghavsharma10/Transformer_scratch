def parseMalformedBamHeader(headerDict):
    """
    Parses the (probably) intended values out of the specified
    BAM header dictionary, which is incompletely parsed by pysam.
    This is caused by some tools incorrectly using spaces instead
    of tabs as a seperator.
    """
    headerString = " ".join(
        "{}:{}".format(k, v) for k, v in headerDict.items() if k != 'CL')
    ret = {}
    for item in headerString.split():
        key, value = item.split(":", 1)
        # build up dict, casting everything back to original type
        ret[key] = type(headerDict.get(key, ""))(value)
    if 'CL' in headerDict:
        ret['CL'] = headerDict['CL']
    return ret