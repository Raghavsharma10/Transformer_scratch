def makedoedict(str1):
    """makedoedict"""
    blocklist = str1.split('..')
    blocklist = blocklist[:-1]#remove empty item after last '..'
    blockdict = {}
    belongsdict = {}
    for num in range(0, len(blocklist)):
        blocklist[num] = blocklist[num].strip()
        linelist = blocklist[num].split(os.linesep)
        aline = linelist[0]
        alinelist = aline.split('=')
        name = alinelist[0].strip()
        aline = linelist[1]
        alinelist = aline.split('=')
        belongs = alinelist[-1].strip()
        theblock = blocklist[num] + os.linesep + '..' + os.linesep + os.linesep
            #put the '..' back in the block
        blockdict[name] = theblock
        belongsdict[name] = belongs
    return [blockdict, belongsdict]