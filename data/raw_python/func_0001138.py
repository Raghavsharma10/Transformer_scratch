def write(file_name, rows, header=None, *args, **kwargs):
        ''' Write rows data to a CSV file (with or without header) '''
        warnings.warn("chirptext.io.CSV is deprecated and will be removed in near future.", DeprecationWarning)
        write_csv(file_name, rows, fieldnames=header, *args, **kwargs)