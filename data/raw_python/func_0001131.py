def process_file(path, processor, encoding='utf-8', mode='rt', *args, **kwargs):
    ''' Process a text file's content. If the file name ends with .gz, read it as gzip file '''
    if mode not in ('rU', 'rt', 'rb', 'r'):
        raise Exception("Invalid file reading mode")
    with open(path, encoding=encoding, mode=mode, *args, **kwargs) as infile:
        return processor(infile)