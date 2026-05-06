def ex(mt, x):
    """ ex : Returns the curtate expectation of life. Life expectancy """
    sum1 = 0
    for j in mt.lx[x + 1:-1]:
        sum1 += j
        #print sum1
    try:
        return sum1 / mt.lx[x] + 0.5
    except:
        return 0