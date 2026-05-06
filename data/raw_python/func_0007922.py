def _getActivePassive(obj1, obj2):
    """ Returns which is the active and the passive objects. """
    speed1 = abs(obj1.lonspeed) if obj1.isPlanet() else -1.0
    speed2 = abs(obj2.lonspeed) if obj2.isPlanet() else -1.0
    if speed1 > speed2:
        return {
            'active': obj1,
            'passive': obj2
        }
    else:
        return {
            'active': obj2,
            'passive': obj1
        }