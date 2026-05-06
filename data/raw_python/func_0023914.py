def grv(struct, position):
    '''
    This function helps to convert date information for showing proper filtering
    '''
    if position == 'year':
        size = 4
    else:
        size = 2

    if (struct[position][2]):
        rightnow = str(struct[position][0]).zfill(size)
    else:
        if position == 'year':
            rightnow = '____'
        else:
            rightnow = '__'
    return rightnow