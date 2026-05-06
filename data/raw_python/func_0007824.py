def light(obj, sun):
    """ Returns if an object is augmenting or diminishing light. """
    dist = angle.distance(sun.lon, obj.lon)
    faster = sun if sun.lonspeed > obj.lonspeed else obj
    if faster == sun:
        return LIGHT_DIMINISHING if dist < 180 else LIGHT_AUGMENTING
    else:
        return LIGHT_AUGMENTING if dist < 180 else LIGHT_DIMINISHING