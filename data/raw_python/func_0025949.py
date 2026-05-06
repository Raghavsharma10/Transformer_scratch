def amod(a, b):
    '''Modulus function which returns numerator if modulus is zero'''
    modded = int(a % b)
    return b if modded is 0 else modded