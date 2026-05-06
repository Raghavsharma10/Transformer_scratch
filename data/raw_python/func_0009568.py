def averageSameExpTimes(imgs_path):
    '''
    average background images with same exposure time
    '''
    firsts = imgs_path[:2]
    imgs = imgs_path[2:]
    for n, i in enumerate(firsts):
        firsts[n] = np.asfarray(imread(i))
    d = DarkCurrentMap(firsts)
    for i in imgs:
        i = imread(i)
        d.addImg(i)
    return d.map()