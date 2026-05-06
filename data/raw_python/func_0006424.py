def get_iso_time():
    '''returns time as ISO string, mapping to and from datetime in ugly way

    convert to string with str()
    '''
    t1 = time.time()
    t2 = datetime.datetime.fromtimestamp(t1)
    t4 = t2.__str__()
    try:
        t4a, t4b = t4.split(".", 1)
    except ValueError:
        t4a = t4
        t4b = '000000'
    t5 = datetime.datetime.strptime(t4a, "%Y-%m-%d %H:%M:%S")
    ms = int(t4b.ljust(6, '0')[:6])
    return t5.replace(microsecond=ms)