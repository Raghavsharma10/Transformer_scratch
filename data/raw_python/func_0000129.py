def cduffy(z, M, vir='200crit', relaxed=True):
    """ NFW conc from Duffy 08 Table 1 for halo mass and redshift"""

    if(vir == '200crit'):
        if relaxed:
            params = [6.71, -0.091, -0.44]
        else:
            params = [5.71, -0.084, -0.47]
    elif(vir == 'tophat'):
        if relaxed:
            params = [9.23, -0.090, -0.69]
        else:
            params = [7.85, -0.081, -0.71]
    elif(vir == '200mean'):
        if relaxed:
            params = [11.93, -0.090, -0.99]
        else:
            params = [10.14, -0.081, -1.01]
    else:
        print("Didn't recognise the halo boundary definition provided %s"
              % (vir))

    return(params[0] * ((M/(2e12/0.72))**params[1]) * ((1+z)**params[2]))