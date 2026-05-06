def get_event_position(voevent, index=0):
    """Extracts the `AstroCoords` from a given `WhereWhen.ObsDataLocation`.

    Note that a packet may include multiple 'ObsDataLocation' entries
    under the 'WhereWhen' section, for example giving locations of an object
    moving over time. Most packets will have only one, however, so the
    default is to just return co-ords extracted from the first.

    Args:
        voevent (:class:`voeventparse.voevent.Voevent`): Root node of the
            VOEvent etree.
        index (int): Index of the ObsDataLocation to extract AstroCoords from.

    Returns:
        Position (:py:class:`.Position2D`): The sky position defined in the
        ObsDataLocation.
    """
    od = voevent.WhereWhen.ObsDataLocation[index]
    ac = od.ObservationLocation.AstroCoords
    ac_sys = voevent.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoordSystem
    sys = ac_sys.attrib['id']

    if hasattr(ac.Position2D, "Name1"):
        assert ac.Position2D.Name1 == 'RA' and ac.Position2D.Name2 == 'Dec'
    posn = Position2D(ra=float(ac.Position2D.Value2.C1),
                      dec=float(ac.Position2D.Value2.C2),
                      err=float(ac.Position2D.Error2Radius),
                      units=ac.Position2D.attrib['unit'],
                      system=sys)
    return posn