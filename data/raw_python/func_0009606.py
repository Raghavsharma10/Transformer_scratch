def linearToPolar(img, center=None,
                  final_radius=None,
                  initial_radius=None,
                  phase_width=None,
                  interpolation=cv2.INTER_AREA, maps=None,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT, **opts):
    '''
    map a 2d (x,y) Cartesian array to a polar (r, phi) array
    using opencv.remap
    '''
    if maps is None:
        mapY, mapX = linearToPolarMaps(img.shape[:2], center, final_radius,
                                       initial_radius, phase_width)
    else:
        mapY, mapX = maps

    o = {'interpolation': interpolation,
         'borderValue': borderValue,
         'borderMode': borderMode}
    o.update(opts)

    return cv2.remap(img, mapY, mapX, **o)