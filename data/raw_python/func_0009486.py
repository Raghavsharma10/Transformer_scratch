def rotate(image, angle, interpolation=cv2.INTER_CUBIC,
           borderMode=cv2.BORDER_REFLECT, borderValue=0):
    '''
    angle [deg]
    '''
    s0, s1 = image.shape
    image_center = (s0 - 1) / 2., (s1 - 1) / 2.
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape,
                            flags=interpolation, borderMode=borderMode,
                            borderValue=borderValue)
    return result