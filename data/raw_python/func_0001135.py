def read_csv_iter(path, fieldnames=None, sniff=True, mode='rt', encoding='utf-8', *args, **kwargs):
    ''' Iterate through CSV rows in a file.
    By default, csv.reader() will be used any output will be a list of lists.
    If fieldnames is provided, DictReader will be used and output will be list of OrderedDict instead.
    CSV sniffing (dialect detection) is enabled by default, set sniff=False to switch it off.
    '''
    with open(path, mode=mode, encoding=encoding) as infile:
        for row in iter_csv_stream(infile, fieldnames=fieldnames, sniff=sniff, *args, **kwargs):
            yield row