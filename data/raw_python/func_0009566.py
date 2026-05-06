def fastMean(img, f=10, inplace=False):
    '''
    for bigger ksizes it if often faster to resize an image
    rather than blur it...
    '''
    s0,s1 = img.shape[:2]
    ss0 = int(round(s0/f))
    ss1 = int(round(s1/f))

    small = cv2.resize(img,(ss1,ss0), interpolation=cv2.INTER_AREA)
        #bigger
    k = {'interpolation':cv2.INTER_LINEAR}
    if inplace:
        k['dst']=img
    return cv2.resize(small,(s1,s0), **k)