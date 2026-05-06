def get_date(layer):
    """
    Returns a custom date representation. A date can be detected or from metadata.
    It can be a range or a simple date in isoformat.
    """
    date = None
    sign = '+'
    date_type = 1
    layer_dates = layer.get_layer_dates()
    # we index the first date!
    if layer_dates:
        sign = layer_dates[0][0]
        date = layer_dates[0][1]
        date_type = layer_dates[0][2]
    if date is None:
        date = layer.created
    # layer date > 2300 is invalid for sure
    # TODO put this logic in date miner
    if date.year > 2300:
        date = None
    if date_type == 0:
        date_type = "Detected"
    if date_type == 1:
        date_type = "From Metadata"
    return get_solr_date(date, (sign == '-')), date_type