def patCrossLines(s0):
    '''make line pattern'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    col = 255
    t = int(s0/100.)
    for pos in np.logspace(0.01,1,10):
        pos = int(round((pos-0.5)*s0/10.))
        cv2.line(arr, (0,pos), (s0,pos), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )
        cv2.line(arr, (pos,0), (pos,s0), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )

    return arr.astype(float)