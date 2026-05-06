def getDarkCurrentAverages(exposuretimes, imgs):
    '''
    return exposure times, image averages for each exposure time
    '''
    x, imgs_p = sortForSameExpTime(exposuretimes, imgs)
    s0, s1 = imgs[0].shape

    imgs = np.empty(shape=(len(x), s0, s1),
                    dtype=imgs[0].dtype)
    for i, ip in zip(imgs, imgs_p):
        if len(ip) == 1:
            i[:] = ip[0]
        else:
            i[:] = averageSameExpTimes(ip)
    return x, imgs