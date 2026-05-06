def annuity(mt, x, n, p, m=1 , *args):
    """Syntax: annuity(nt, x, n, p, m, ['a/g', q], -d)
    Args:
        mt = the mortality table
        x = the age as integer number.   
        n = A integer number (term of insurance in years) or 'w' = whole-life. 
            (Also, 99 years is defined to be whole-life).
        p = Moment of payment. Syntaxis: 0 = begining of each period (prepaid), 1 = end of each period (postpaid),
    Optional variables:
        m = Payable 'm' per year (frational payments). Default = 1 (annually)
        a or g = a: Arithmetical / g: Geometrical
        q = The increase rate. Syntax: ['g',q] or ['a',q]. For example, ['g',0.03]
    Deferring period:
        -d = The n-years deferring period as negative number. 
    """
    l = len(args)
    post = False
    incr = False
    deff = False
    arit = False
    wh_l = False

    if isinstance(n,str) or n == 99:
        wh_l = True
    else:
        pass

    if isinstance(m,int) and m >=0 and l == 0: 
        pass
    elif l == 0 and isinstance(m,list):  
        args = (m,)
        m = 1
        incr = True
    elif l == 0 and int(m) < 0:
        args = False
        deff = True
        t = int(m) * -1
        m = 1
    elif l == 1:
        if isinstance(args[0], list):
            incr = True
        elif isinstance(args[0], int):
            if isinstance(m, list):
                deff = True
                incr = True
                t = int(args[0]) * -1
                args = (m, )
                m = 1
            else:
                deff = True
                t = int(args[0]) * -1
                args = False
        else:
            pass
    elif l == 2:        
        if isinstance(args[0], list):
            deff = True
            t = int(args[1]) * -1
            incr = True
        elif isinstance(args[0], int):
            deff = True
            t = int(args[0]) * -1
            args = args[1]
        else:
            pass
    else:        
        pass
    
    if p == 1:
        post = True
    elif p == 0:
        pass
    else:
        print('Error: payment value is 0 or 1')

    if incr:
        if 'a' in args[0]:
            arit = True
            incr = False
        elif 'g' in args[0]:
            incr = True
            q = args[0][1]
        else:
            return "Error: increasing value is 'a' or 'g'"

    else:
        pass

    if not incr and not deff and not wh_l and not post:
        return aaxn(mt, x, n, m)
    elif not incr and not deff and not wh_l and post:
        return axn(mt, x, n, m)
    elif not incr and not deff and wh_l and not post:
        return aax(mt, x, m)
    elif not incr and not deff and wh_l and post:
        return ax(mt, x, m)
    elif not incr and deff and not wh_l and not post:
        return taaxn(mt, x, n, t, m)
    elif not incr and deff and not wh_l and post:
        return taxn(mt, x, n, t, m)
    elif not incr and deff and wh_l and not post:
        return taax(mt, x, t, m)
    elif not incr and deff and wh_l and post:
        return tax(mt, x, t, m)
    elif incr and not deff and not wh_l and not post:
        return qaaxn(mt, x, n, q, m)
    elif incr and not deff and not wh_l and post:
        return qaxn(mt, x, n, q, m)
    elif incr and not deff and wh_l and not post:
        return qaax(mt, x, q, m)    
    elif incr and not deff and wh_l and post:
        return qax(mt, x, q, m)
    elif incr and deff and not wh_l and not post:
        return qtaaxn(mt, x, n, t, q, m)
    elif incr and deff and not wh_l and post:
        return qtaxn(mt, x, n, t, q, m)
    elif incr and deff and wh_l and not post:
        return qtaax(mt, x, t, q, m)
    else:
        #elif incr and deff and wh_l and post:
        return Itax(mt, x, t)