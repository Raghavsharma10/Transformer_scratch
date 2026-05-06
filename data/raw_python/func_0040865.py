def parse_vmstat(text):
    ''' Parse vmstat output. '''
    lines = text.splitlines()
    results = Info()  # TODO use MemInfo

    try:
        PAGESIZE = int(lines[0].split()[-2])
    except IndexError:
        PAGESIZE = 4096

    for line in lines[1:]:      # dump header
        if not line[0] == 80:   # b'P' startswith Page...
            break
        tokens = line.split()
        name, value = tokens[1][:-1].decode('ascii'), tokens[-1][:-1]
        results[name] = int(value) * PAGESIZE

    return results