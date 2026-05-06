def modifiedLaplacian(img):
    ''''LAPM' algorithm (Nayar89)'''
    M = np.array([-1, 2, -1])
    G = cv2.getGaussianKernel(ksize=3, sigma=-1)
    Lx = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=M, kernelY=G)
    Ly = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=G, kernelY=M)
    FM = np.abs(Lx) + np.abs(Ly)
    return cv2.mean(FM)[0]