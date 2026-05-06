def t_NUMBER(t):
    r'[0-9]+[KkMmGg]?'
    exponents = {
            'G' : 30, 'g' : 30,
            'M' : 20, 'm' : 20,
            'K' : 10, 'k' : 10,
            }
    if t.value[-1] in exponents:
        t.value = math.ldexp(int(t.value[:-1]), exponents[t.value[-1]])
    else:
        t.value = int(t.value)
    return t