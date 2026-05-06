def putTextAlpha(img, text, alpha, org, fontFace, fontScale, color,
                 thickness):  # , lineType=None
    '''
    Extends cv2.putText with [alpha] argument
    '''

    x, y = cv2.getTextSize(text, fontFace,
                           fontScale, thickness)[0]

    ox, oy = org

    imgcut = img[oy - y - 3:oy, ox:ox + x]

    if img.ndim == 3:
        txtarr = np.zeros(shape=(y + 3, x, 3), dtype=np.uint8)
    else:
        txtarr = np.zeros(shape=(y + 3, x), dtype=np.uint8)

    cv2.putText(txtarr, text, (0, y), fontFace,
                fontScale, color,
                thickness=thickness
                #, lineType=lineType
                )

    cv2.addWeighted(txtarr, alpha, imgcut, 1, 0, imgcut, -1)
    return img