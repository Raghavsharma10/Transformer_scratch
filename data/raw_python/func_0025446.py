def track_storms(storm_objects, times, distance_components, distance_maxima, distance_weights, tracked_objects=None):
    """
    Given the output of extract_storm_objects, this method tracks storms through time and merges individual
    STObjects into a set of tracks.

    Args:
        storm_objects: list of list of STObjects that have not been tracked.
        times: List of times associated with each set of STObjects
        distance_components: list of function objects that make up components of distance function
        distance_maxima: array of maximum values for each distance for normalization purposes
        distance_weights: weight given to each component of the distance function. Should add to 1.
        tracked_objects: List of STObjects that have already been tracked.
    Returns:
        tracked_objects:
    """
    obj_matcher = ObjectMatcher(distance_components, distance_weights, distance_maxima)
    if tracked_objects is None:
        tracked_objects = []
    for t, time in enumerate(times):
        past_time_objects = []
        for obj in tracked_objects:
            if obj.end_time == time - obj.step:
                past_time_objects.append(obj)
        if len(past_time_objects) == 0:
            tracked_objects.extend(storm_objects[t])
        elif len(past_time_objects) > 0 and len(storm_objects[t]) > 0:
            assignments = obj_matcher.match_objects(past_time_objects, storm_objects[t], times[t-1], times[t])
            unpaired = list(range(len(storm_objects[t])))
            for pair in assignments:
                past_time_objects[pair[0]].extend(storm_objects[t][pair[1]])
                unpaired.remove(pair[1])
            if len(unpaired) > 0:
                for up in unpaired:
                    tracked_objects.append(storm_objects[t][up])
    return tracked_objects