def sortForSameExpTime(expTimes, img_paths):  # , excludeSingleImg=True):
    '''
    return image paths sorted for same exposure time
    '''
    d = {}
    for e, i in zip(expTimes, img_paths):
        if e not in d:
            d[e] = []
        d[e].append(i)
#     for key in list(d.keys()):
#         if len(d[key]) == 1:
#             print('have only one image of exposure time [%s]' % key)
#             print('--> exclude that one')
#             d.pop(key)
    d = OrderedDict(sorted(d.items()))
    return list(d.keys()), list(d.values())