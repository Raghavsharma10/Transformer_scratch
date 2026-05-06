def parse_sysctl(text):
    ''' Parse sysctl output. '''
    lines = text.splitlines()
    results = {}
    for line in lines:
        key, _, value = line.decode('ascii').partition(': ')

        if key == 'hw.memsize':
            value = int(value)

        elif key == 'vm.swapusage':
            values = value.split()[2::3]            # every third token
            su_unit = values[0][-1].lower()         # get unit, 'M'
            PAGESIZE = 1024
            if su_unit == 'm':
                PAGESIZE = 1024 * 1024

            value = [ (float(val[:-1]) * PAGESIZE) for val in values ]

        results[key] = value
    return results