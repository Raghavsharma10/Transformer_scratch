def copy(src, trg, transform=None):
    ''' copy items with optional fields transformation
    '''
    source = open(src[0], src[1])
    target = open(trg[0], trg[1], autocommit=1000)

    for item in source.get():
        item = dict(item)
        if '_id' in item:
            del item['_id']

        if transform:
            item = transform(item)
        target.put(trg[0](item))

    source.close()
    target.commit()
    target.close()