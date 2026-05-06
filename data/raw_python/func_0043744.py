def create(*units):
    """create this unit within the game as specified"""
    ret = []
    for unit in units: # implemented using sc2simulator.ScenarioUnit
        x, y = unit.position[:2]
        pt = Point2D(x=x, y=y)
        unit.tag = 0 # forget any tag because a new unit will be created
        new = DebugCommand(create_unit=DebugCreateUnit(
            unit_type   = unit.code,
            owner       = unit.owner,
            pos         = pt,
            quantity    = 1,
        ))
        ret.append(new)
    return ret