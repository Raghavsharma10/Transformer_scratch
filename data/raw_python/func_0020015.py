def _slice_required_len(slice_obj):
    """
    Calculate how many items must be in the collection to satisfy this slice

    returns `None` for slices may vary based on the length of the underlying collection
    such as `lst[-1]` or `lst[::]`
    """
    if slice_obj.step and slice_obj.step != 1:
        return None
    # (None, None, *) requires the entire list
    if slice_obj.start is None and slice_obj.stop is None:
        return None

    # Negative indexes are hard without knowing the collection length
    if slice_obj.start and slice_obj.start < 0:
        return None
    if slice_obj.stop and slice_obj.stop < 0:
        return None

    if slice_obj.stop:
        if slice_obj.start and slice_obj.start > slice_obj.stop:
            return 0
        return slice_obj.stop
    return slice_obj.start + 1