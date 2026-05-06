def distant_level(reference_level, distance, reference_distance=1.0):
    """
    Calculates the sound pressure level
    in dependence of a distance
    where a perfect ball-shaped source and spread is assumed.

    reference_level: Sound pressure level in reference distance in dB
    distance: Distance to calculate sound pressure level for, in meters
    reference_distance: reference distance in meters (defaults to 1)
    """
    rel_dist = float(reference_distance) / float(distance)
    level = float(reference_level) + 20.0 * (math.log(rel_dist) / math.log(10))
    return level