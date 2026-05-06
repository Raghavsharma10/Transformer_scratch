def get_float_time():
    '''returns time as double precision floats - Time64 in pytables - mapping to and from python datetime's

    '''
    t1 = time.time()
    t2 = datetime.datetime.fromtimestamp(t1)
    return time.mktime(t2.timetuple()) + 1e-6 * t2.microsecond