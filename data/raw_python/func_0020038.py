def date_decimal_hook(dct):
    '''The default JSON decoder hook. It is the inverse of
:class:`stdnet.utils.jsontools.JSONDateDecimalEncoder`.'''
    if '__datetime__' in dct:
        return todatetime(dct['__datetime__'])
    elif '__date__' in dct:
        return todatetime(dct['__date__']).date()
    elif '__decimal__' in dct:
        return Decimal(dct['__decimal__'])
    else:
        return dct