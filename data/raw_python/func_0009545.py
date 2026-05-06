def shiftImage(u, v, t, img, interpolation=cv2.INTER_LANCZOS4):
    '''
    remap an image using velocity field
    '''
    ny,nx = u.shape
    sy, sx = np.mgrid[:float(ny):1,:float(nx):1]
    sx += u*t
    sy += v*t
    return cv2.remap(img.astype(np.float32), 
                    (sx).astype(np.float32),
                    (sy).astype(np.float32), interpolation)