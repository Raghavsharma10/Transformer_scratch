def read_csv(path, fieldnames=None, sniff=True, encoding='utf-8', *args, **kwargs):
    ''' Read CSV rows as table from a file.
    By default, csv.reader() will be used any output will be a list of lists.
    If fieldnames is provided, DictReader will be used and output will be list of OrderedDict instead.
    CSV sniffing (dialect detection) is enabled by default, set sniff=False to switch it off.
    '''
    return list(r for r in read_csv_iter(path, fieldnames=fieldnames, sniff=sniff, encoding=encoding, *args, **kwargs))