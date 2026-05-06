def pluralize(data_type):
    """
    adds s to the data type or the correct english plural form
    """
    known = {
             u"address": u"addresses", 
             u"company": u"companies"
    }
    if data_type in known.keys():
        return known[data_type]
    else:
        return u"%ss" % data_type