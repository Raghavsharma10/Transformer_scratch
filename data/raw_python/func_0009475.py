def vignettingFromRandomSteps(imgs, bg, inPlane_scale_factor=None,
                              debugFolder=None, **kwargs):
    '''
    important: first image should shown most iof the device
    because it is used as reference
    '''
    # TODO: inPlane_scale_factor
    if debugFolder:
        debugFolder = PathStr(debugFolder)

    s = ObjectVignettingSeparation(imgs[0], bg,  **kwargs)
    for img in imgs[1:]:
        fit = s.addImg(img)

        if debugFolder and fit is not False:
            imwrite(debugFolder.join('fit_%s.tiff' % len(s.fits)), fit)

    if debugFolder:
        imwrite(debugFolder.join('init.tiff'), s.flatField)

    smoothed_ff, mask, flatField, obj = s.separate()

    if debugFolder:
        imwrite(debugFolder.join('object.tiff'), obj)
        imwrite(debugFolder.join('flatfield.tiff'), flatField, dtype=float)
        imwrite(debugFolder.join('flatfield_smoothed.tiff'), smoothed_ff,
                dtype=float)

    return smoothed_ff, mask