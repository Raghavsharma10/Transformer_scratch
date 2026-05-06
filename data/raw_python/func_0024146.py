def date_to_solr(d):
    """ converts DD-MM-YYYY to YYYY-MM-DDT00:00:00Z"""
    return "{y}-{m}-{day}T00:00:00Z".format(day=d[:2], m=d[3:5], y=d[6:]) if d else d