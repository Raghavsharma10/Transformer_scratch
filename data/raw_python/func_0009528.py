def patCircles(s0):
    '''make circle array'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    col = 255
    for rad in np.linspace(s0,s0/7.,10):
        cv2.circle(arr, (0,0), int(round(rad)), color=col, 
                   thickness=-1, lineType=cv2.LINE_AA )
        if col:
            col = 0
        else:
            col = 255
            

    return arr.astype(float)