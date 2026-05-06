def longcount_generator(baktun, katun, tun, uinal, kin):
    '''Generate long counts, starting with input'''
    j = to_jd(baktun, katun, tun, uinal, kin)

    while True:
        yield from_jd(j)
        j = j + 1