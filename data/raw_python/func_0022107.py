def layer_pre_save(instance, *args, **kwargs):
    """
    Used to check layer validity.
    """

    is_valid = True

    # we do not need to check validity for WM layers
    if not instance.service.type == 'Hypermap:WorldMap':

        # 0. a layer is invalid if its service its invalid as well
        if not instance.service.is_valid:
            is_valid = False
            LOGGER.debug('Layer with id %s is marked invalid because its service is invalid' % instance.id)

        # 1. a layer is invalid with an extent within (-2, -2, +2, +2)
        if instance.bbox_x0 > -2 and instance.bbox_x1 < 2 and instance.bbox_y0 > -2 and instance.bbox_y1 < 2:
            is_valid = False
            LOGGER.debug(
                'Layer with id %s is marked invalid because its extent is within (-2, -2, +2, +2)' % instance.id
            )

    instance.is_valid = is_valid