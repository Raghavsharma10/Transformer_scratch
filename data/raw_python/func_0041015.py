def read_header(filename):
    ''' returns a dictionary of values in the header of the given file '''
    header = {}
    in_header = False
    data = nl.universal_read(filename)
    lines = [x.strip() for x in data.split('\n')]
    for line in lines:
        if line=="*** Header Start ***":
            in_header=True
            continue
        if line=="*** Header End ***":
            return header
        fields = line.split(": ")
        if len(fields)==2:
            header[fields[0]] = fields[1]