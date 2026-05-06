def writer(f):
    '''CSV writer factory for CADA format'''
    return unicodecsv.writer(f, encoding='utf-8', delimiter=b',', quotechar=b'"')