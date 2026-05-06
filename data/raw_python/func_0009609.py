def varianceOfLaplacian(img):
    ''''LAPV' algorithm (Pech2000)'''
    lap = cv2.Laplacian(img, ddepth=-1)#cv2.cv.CV_64F)
    stdev = cv2.meanStdDev(lap)[1]
    s = stdev[0]**2
    return s[0]