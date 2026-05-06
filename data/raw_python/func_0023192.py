def get_number_unit(number):
    """get the unit of number"""
    n = str(float(number))
    mult, submult = n.split('.')
    if float(submult) != 0:
        unit = '0.' + (len(submult)-1)*'0' + '1'
        return float(unit)
    else:
        return float(1)