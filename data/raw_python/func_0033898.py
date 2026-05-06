def build_segment_list_one(engine, gps_start_time, gps_end_time, ifo, segment_name, version = None, start_pad = 0, end_pad = 0):
    """Builds a list of segments satisfying the given criteria """
    seg_result = segmentlist([])
    sum_result = segmentlist([])

    # Is there any way to get segment and segement summary in one query?
    # Maybe some sort of outer join where we keep track of which segment
    # summaries we've already seen.
    sql = "SELECT segment_summary.start_time, segment_summary.end_time "
    sql += "FROM segment_definer, segment_summary "
    sql += "WHERE segment_summary.segment_def_id = segment_definer.segment_def_id "
    sql += "AND   segment_definer.ifos = '%s' " % ifo
    if engine.__class__ == query_engine.LdbdQueryEngine:
       sql += "AND segment_summary.segment_def_cdb = segment_definer.creator_db "
    sql += "AND   segment_definer.name = '%s' " % segment_name
    sql += "AND   segment_definer.version = %s " % version
    sql += "AND NOT (%s > segment_summary.end_time OR segment_summary.start_time > %s)" % (gps_start_time, gps_end_time)

    rows = engine.query(sql)

    for sum_start_time, sum_end_time in rows:
        sum_start_time = (sum_start_time < gps_start_time) and gps_start_time or sum_start_time
        sum_end_time = (sum_end_time > gps_end_time) and gps_end_time or sum_end_time

        sum_result |= segmentlist([segment(sum_start_time, sum_end_time)])

    # We can't use queries paramaterized with ? since the ldbd protocol doesn't support it...
    sql = "SELECT segment.start_time + %d, segment.end_time + %d " % (start_pad, end_pad)
    sql += "FROM segment, segment_definer "
    sql += "WHERE segment.segment_def_id = segment_definer.segment_def_id "

    if engine.__class__ == query_engine.LdbdQueryEngine:
       sql += "AND segment.segment_def_cdb = segment_definer.creator_db "
    sql += "AND   segment_definer.ifos = '%s' " % ifo
    sql += "AND   segment_definer.name = '%s' " % segment_name
    sql += "AND   segment_definer.version = %s " % version
    sql += "AND NOT (%s > segment.end_time OR segment.start_time > %s)" % (gps_start_time, gps_end_time)

    rows = engine.query(sql)
    
    for seg_start_time, seg_end_time in rows:
        seg_start_time = (seg_start_time < gps_start_time) and gps_start_time or seg_start_time
        seg_end_time = (seg_end_time > gps_end_time) and gps_end_time or seg_end_time

        seg_result |= segmentlist([segment(seg_start_time, seg_end_time)])

    engine.close()

    return sum_result, seg_result