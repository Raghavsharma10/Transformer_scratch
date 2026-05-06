def equalizeImage(img, save_path=None, name_additive='_eqHist'):
    '''
    Equalize the histogram (contrast) of an image
    works with RGB/multi-channel images
    and flat-arrays

    @param img  - image_path or np.array
    @param save_path if given output images will be saved there
    @param name_additive if given this additive will be appended to output images

    @return output images if input images are numpy.arrays and no save_path is given
    @return None elsewise
    '''

    if isinstance(img, string_types):
        img = PathStr(img)
        if not img.exists():
            raise Exception("image path doesn't exist")
        img_name = img.basename().replace('.', '%s.' % name_additive)
        if save_path is None:
            save_path = img.dirname()
        img = cv2.imread(img)

    if img.dtype != np.dtype('uint8'):
        # openCV cannot work with float arrays or uint > 8bit
        eqFn = _equalizeHistogram
    else:
        eqFn = cv2.equalizeHist
    if len(img.shape) == 3:  # multi channel img like rgb
        for i in range(img.shape[2]):
            img[:, :, i] = eqFn(img[:, :, i])
    else:  # grey scale image
        img = eqFn(img)
    if save_path:
        img_name = PathStr(save_path).join(img_name)
        cv2.imwrite(img_name, img)
    return img