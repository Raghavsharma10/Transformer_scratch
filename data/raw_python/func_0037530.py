def get_event_time_as_utc(voevent, index=0):
    """
    Extracts the event time from a given `WhereWhen.ObsDataLocation`.

    Returns a datetime (timezone-aware, UTC).

    Accesses a `WhereWhere.ObsDataLocation.ObservationLocation`
    element and returns the AstroCoords.Time.TimeInstant.ISOTime element,
    converted to a (UTC-timezoned) datetime.

    Note that a packet may include multiple 'ObsDataLocation' entries
    under the 'WhereWhen' section, for example giving locations of an object
    moving over time. Most packets will have only one, however, so the
    default is to access the first.

    This function now implements conversion from the
    TDB (Barycentric Dynamical Time) time scale in ISOTime format,
    since this is the format used by GAIA VOEvents.
    (See also http://docs.astropy.org/en/stable/time/#time-scale )

    Other timescales (i.e. TT, GPS) will presumably be formatted as a
    TimeOffset, parsing this format is not yet implemented.

    Args:
        voevent (:class:`voeventparse.voevent.Voevent`): Root node of the VOevent
            etree.
        index (int): Index of the ObsDataLocation to extract an ISOtime from.

    Returns:
        :class:`datetime.datetime`: Datetime representing the event-timestamp,
        converted to UTC (timezone aware).

    """
    try:
        od = voevent.WhereWhen.ObsDataLocation[index]
        ol = od.ObservationLocation
        coord_sys = ol.AstroCoords.attrib['coord_system_id']
        timesys_identifier = coord_sys.split('-')[0]

        if timesys_identifier == 'UTC':
            isotime_str = str(ol.AstroCoords.Time.TimeInstant.ISOTime)
            return iso8601.parse_date(isotime_str)
        elif (timesys_identifier == 'TDB'):
            isotime_str = str(ol.AstroCoords.Time.TimeInstant.ISOTime)
            isotime_dtime = iso8601.parse_date(isotime_str)
            tdb_time = astropy.time.Time(isotime_dtime, scale='tdb')
            return tdb_time.utc.to_datetime().replace(tzinfo=pytz.UTC)
        elif (timesys_identifier == 'TT' or timesys_identifier == 'GPS'):
            raise NotImplementedError(
                "Conversion from time-system '{}' to UTC not yet implemented"
            )
        else:
            raise ValueError(
                'Unrecognised time-system: {} (badly formatted VOEvent?)'.format(
                    timesys_identifier
                )
            )

    except AttributeError:
        return None