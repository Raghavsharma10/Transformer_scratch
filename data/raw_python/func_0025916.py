def haab_monthcalendar(baktun=None, katun=None, tun=None, uinal=None, kin=None, jdc=None):
    '''For a given long count, return a calender of the current haab month, divided into tzolkin "weeks"'''
    if not jdc:
        jdc = to_jd(baktun, katun, tun, uinal, kin)

    haab_number, haab_month = to_haab(jdc)
    first_j = jdc - haab_number + 1

    tzolkin_start_number, tzolkin_start_name = to_tzolkin(first_j)

    gen_longcount = longcount_generator(*from_jd(first_j))
    gen_tzolkin = tzolkin_generator(tzolkin_start_number, tzolkin_start_name)

    # 13 day long tzolkin 'weeks'
    lpad = tzolkin_start_number - 1
    rpad = 13 - (tzolkin_start_number + 19 % 13)

    monlen = month_length(haab_month)

    days = [None] * lpad + list(range(1, monlen + 1)) + rpad * [None]

    def g(x, generate):
        if x is None:
            return None
        return next(generate)

    return [[(k, g(k, gen_tzolkin), g(k, gen_longcount)) for k in days[i:i + 13]] for i in range(0, len(days), 13)]