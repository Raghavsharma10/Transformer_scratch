def parse_solr_time_range_as_pair(time_filter):
    """
    :param time_filter: [2013-03-01 TO 2013-05-01T00:00:00]
    :return: (2013-03-01, 2013-05-01T00:00:00)
    """
    pattern = "\\[(.*) TO (.*)\\]"
    matcher = re.search(pattern, time_filter)
    if matcher:
        return matcher.group(1), matcher.group(2)
    else:
        raise Exception("Regex {0} couldn't parse {1}".format(pattern, time_filter))