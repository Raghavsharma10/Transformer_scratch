def sunRelation(obj, sun):
    """ Returns an object's relation with the sun. """
    if obj.id == const.SUN:
        return None
    dist = abs(angle.closestdistance(sun.lon, obj.lon))
    if dist < 0.2833: return CAZIMI
    elif dist < 8.0: return COMBUST
    elif dist < 16.0: return UNDER_SUN
    else:
        return None