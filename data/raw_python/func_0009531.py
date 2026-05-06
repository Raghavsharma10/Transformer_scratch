def patSiemensStar(s0, n=72, vhigh=255, vlow=0, antiasing=False):
    '''make line pattern'''
    arr = np.full((s0,s0),vlow, dtype=np.uint8)
    c = int(round(s0/2.))
    s = 2*np.pi/(2*n)
    step =  0
    for i in range(2*n): 
        p0 = round(c+np.sin(step)*2*s0)
        p1 = round(c+np.cos(step)*2*s0)
       
        step += s

        p2 = round(c+np.sin(step)*2*s0)
        p3 = round(c+np.cos(step)*2*s0)

        pts = np.array(((c,c), 
                        (p0,p1),
                        (p2,p3) ), dtype=int)

        cv2.fillConvexPoly(arr, pts,
                           color=vhigh if i%2 else vlow, 
                           lineType=cv2.LINE_AA  if antiasing else 0)
    arr[c,c]=0
    
    return arr.astype(float)