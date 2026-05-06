def missingkeys_nonstandard(block, commdct, dtls, objectlist, afield='afiled %s'):
    """This is an object list where thre is no first field name
    to give a hint of what the first field name should be"""
    afield = 'afield %s'
    for key_txt in objectlist:
        key_i = dtls.index(key_txt.upper())
        comm = commdct[key_i]
        if block:
            blk = block[key_i]
        for i, cmt in enumerate(comm):
            if cmt == {}:
                first_i = i
                break
        for i, cmt in enumerate(comm):
            if i >= first_i:
                if block:
                    comm[i]['field'] = ['%s' % (blk[i])]
                else:
                    comm[i]['field'] = [afield % (i - first_i + 1,),]