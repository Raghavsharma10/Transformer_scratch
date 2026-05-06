def reader(f):
    '''CSV Reader factory for CADA format'''
    return unicodecsv.reader(f, encoding='utf-8', delimiter=b',', quotechar=b'"')