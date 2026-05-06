def compare_provenance(
        this_provenance, other_provenance,
        left_outer_diff = "In current but not comparison",
        right_outer_diff = "In comparison but not current"):
    """Utility function to compare two abritrary provenance dicts
    returns number of discrepancies.

    Parameters
    ----------
    this_provenance: provenance dict (to be compared to "other_provenance")
    other_provenance: comparison provenance dict

    (optional)
    left_outer_diff: description/prefix used when printing items in this_provenance but not in other_provenance
    right_outer_diff: description/prefix used when printing items in other_provenance but not in this_provenance

    Returns
    -----------
    Number of discrepancies (0: None)
    """
    ## if either this or other items is null, return 0
    if (not this_provenance or not other_provenance):
        return 0

    this_items = set(this_provenance.items())
    other_items = set(other_provenance.items())

    # Two-way diff: are any modules introduced, and are any modules lost?
    new_diff = this_items.difference(other_items)
    old_diff = other_items.difference(this_items)
    warn_str = ""
    if len(new_diff) > 0:
        warn_str += "%s: %s" % (
            left_outer_diff,
            _provenance_str(new_diff))
    if len(old_diff) > 0:
        warn_str += "%s: %s" % (
            right_outer_diff,
            _provenance_str(old_diff))

    if len(warn_str) > 0:
        warnings.warn(warn_str, Warning)

    return(len(new_diff)+len(old_diff))