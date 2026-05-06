def textbetween(variable,
                firstnum=None,
                secondnum=None,
                locationoftext='regular'):
    """
    Get The Text Between Two Parts
    """
    if locationoftext == 'regular':
        return variable[firstnum:secondnum]
    elif locationoftext == 'toend':
        return variable[firstnum:]
    elif locationoftext == 'tostart':
        return variable[:secondnum]