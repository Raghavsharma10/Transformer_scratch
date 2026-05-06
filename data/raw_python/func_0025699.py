def qs_from_dict(qsdict, prefix=""):
    ''' Same as dict_from_qs, but in reverse
        i.e. {"period": {"di": {}, "fhr": {}}} => "period.di,period.fhr"
    '''
    prefix = prefix + '.' if prefix else ""

    def descend(qsd):
        for key, val in sorted(qsd.items()):
            if val:
                yield qs_from_dict(val, prefix + key)
            else:
                yield prefix + key
    return ",".join(descend(qsdict))