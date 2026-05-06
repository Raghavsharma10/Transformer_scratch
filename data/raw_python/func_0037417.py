def add_where_when(voevent, coords, obs_time, observatory_location,
                   allow_tz_naive_datetime=False):
    """
    Add details of an observation to the WhereWhen section.

    We

    Args:
        voevent(:class:`Voevent`): Root node of a VOEvent etree.
        coords(:class:`.Position2D`): Sky co-ordinates of event.
        obs_time(datetime.datetime): Nominal DateTime of the observation. Must
            either be timezone-aware, or should be carefully verified as
            representing UTC and then set parameter
            ``allow_tz_naive_datetime=True``.
        observatory_location(str): Telescope locale, e.g. 'La Palma'.
            May be a generic location as listed under
            :class:`voeventparse.definitions.observatory_location`.
        allow_tz_naive_datetime (bool): (Default False). Accept timezone-naive
            datetime-timestamps. See comments for ``obs_time``.

    """

    # .. todo:: Implement TimeError using datetime.timedelta
    if obs_time.tzinfo is not None:
        utc_naive_obs_time = obs_time.astimezone(pytz.utc).replace(tzinfo=None)
    elif not allow_tz_naive_datetime:
        raise ValueError(
            "Datetime passed without tzinfo, cannot be sure if it is really a "
            "UTC timestamp. Please verify function call and either add tzinfo "
            "or pass parameter 'allow_tz_naive_obstime=True', as appropriate",
        )
    else:
        utc_naive_obs_time = obs_time

    obs_data = etree.SubElement(voevent.WhereWhen, 'ObsDataLocation')
    etree.SubElement(obs_data, 'ObservatoryLocation', id=observatory_location)
    ol = etree.SubElement(obs_data, 'ObservationLocation')
    etree.SubElement(ol, 'AstroCoordSystem', id=coords.system)
    ac = etree.SubElement(ol, 'AstroCoords',
                          coord_system_id=coords.system)
    time = etree.SubElement(ac, 'Time', unit='s')
    instant = etree.SubElement(time, 'TimeInstant')
    instant.ISOTime = utc_naive_obs_time.isoformat()
    # iso_time = etree.SubElement(instant, 'ISOTime') = obs_time.isoformat()

    pos2d = etree.SubElement(ac, 'Position2D', unit=coords.units)
    pos2d.Name1 = 'RA'
    pos2d.Name2 = 'Dec'
    pos2d_val = etree.SubElement(pos2d, 'Value2')
    pos2d_val.C1 = coords.ra
    pos2d_val.C2 = coords.dec
    pos2d.Error2Radius = coords.err