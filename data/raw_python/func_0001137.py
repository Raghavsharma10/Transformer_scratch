def write_csv(path, rows, dialect='excel', fieldnames=None, quoting=csv.QUOTE_ALL, extrasaction='ignore', *args, **kwargs):
        ''' Write rows data to a CSV file (with or without fieldnames) '''
        if not quoting:
            quoting = csv.QUOTE_MINIMAL
        if 'lineterminator' not in kwargs:
            kwargs['lineterminator'] = '\n'  # use \n to fix double-line in Windows
        with open(path, mode='wt', newline='') as csvfile:
            if fieldnames:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect=dialect, quoting=quoting, extrasaction=extrasaction, *args, **kwargs)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            else:
                writer = csv.writer(csvfile, dialect=dialect, quoting=quoting, *args, **kwargs)
                for row in rows:
                    writer.writerow(row)