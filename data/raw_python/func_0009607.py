def polarToLinear(img, shape=None, center=None, maps=None,
                  interpolation=cv2.INTER_AREA,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT, **opts):
    '''
    map a 2d polar (r, phi) polar array to a  Cartesian (x,y) array
    using opencv.remap
    '''

    if maps is None:
        mapY, mapX = polarToLinearMaps(img.shape[:2], shape, center)
    else:
        mapY, mapX = maps

    o = {'interpolation': interpolation,
         'borderValue': borderValue,
         'borderMode': borderMode}
    o.update(opts)

    return cv2.remap(img, mapY, mapX, **o)