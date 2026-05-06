def modify(*units):
    """set the unit defined by in-game tag with desired properties
    NOTE: all units must be owned by the same player or the command fails."""
    ret = []
    for unit in units: # add one command for each attribute
        for attr, idx in [("energy", 1), ("life", 2), ("shields", 3)]: # see debug_pb2.UnitValue for enum declaration
            newValue = getattr(unit, attr)
            if not newValue: continue # don't bother setting something that isn't necessary
            new = DebugCommand(unit_value=DebugSetUnitValue(
                value       = newValue,
                unit_value  = idx,
                unit_tag    = unit.tag))
            ret.append(new)
    return ret