def normalizedGraylevelVariance(img):
    ''''GLVN' algorithm (Santos97)'''
    mean, stdev = cv2.meanStdDev(img)
    s = stdev[0]**2 / mean[0]
    return s[0]