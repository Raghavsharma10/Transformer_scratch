def patText(s0):
    '''make text pattern'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    s = int(round(s0/100.))
    p1 = 0
    pp1 = int(round(s0/10.))
    for pos0 in np.linspace(0,s0,10):
        cv2.putText(arr, 'helloworld', (p1,int(round(pos0))), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=s,
                    color=255, thickness=s,
                    lineType=cv2.LINE_AA )
        if p1:
            p1 = 0
        else:
            p1 = pp1
    return arr.astype(float)