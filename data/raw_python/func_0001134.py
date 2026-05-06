def iter_csv_stream(input_stream, fieldnames=None, sniff=False, *args, **kwargs):
    ''' Read CSV content as a table (list of lists) from an input stream '''
    if 'dialect' not in kwargs and sniff:
        kwargs['dialect'] = csv.Sniffer().sniff(input_stream.read(1024))
        input_stream.seek(0)
    if 'quoting' in kwargs and kwargs['quoting'] is None:
        kwargs['quoting'] = csv.QUOTE_MINIMAL
    if fieldnames:
        # read csv using dictreader
        if isinstance(fieldnames, bool):
            reader = csv.DictReader(input_stream, *args, **kwargs)
        else:
            reader = csv.DictReader(input_stream, *args, fieldnames=fieldnames, **kwargs)
        for row in reader:
            yield row
    else:
        csvreader = csv.reader(input_stream, *args, **kwargs)
        for row in csvreader:
            yield row