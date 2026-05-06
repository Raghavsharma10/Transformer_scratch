def build_segment_list(engine, gps_start_time, gps_end_time, ifo, segment_name, version = None, start_pad = 0, end_pad = 0):
    """Optains a list of segments for the given ifo, name and version between the
    specified times.  If a version is given the request is straightforward and is
    passed on to build_segment_list_one.  Otherwise more complex processing is
    performed (not yet implemented)"""
    if version is not None:
        return build_segment_list_one(engine, gps_start_time, gps_end_time, ifo, segment_name, version, start_pad, end_pad)

    # This needs more sophisticated logic, for the moment just return the latest
    # available version
    sql  = "SELECT max(version) FROM segment_definer "
    sql += "WHERE  segment_definer.ifos = '%s' " % ifo
    sql += "AND   segment_definer.name = '%s' " % segment_name

    rows = engine.query(sql)
    version = len(rows[0]) and rows[0][0] or 1

    return build_segment_list_one(engine, gps_start_time, gps_end_time, ifo, segment_name, version, start_pad, end_pad)