def run_query_segments(doc, proc_id, engine, gps_start_time, gps_end_time, included_segments_string, excluded_segments_string = None, write_segments = True, start_pad = 0, end_pad = 0):
    """Runs a segment query.  This was originally part of ligolw_query_segments, but now is also
    used by ligolw_segments_from_cats.

    The write_segments option is provided so callers can coalesce segments obtained over
    sever invocations (as segments_from_cats does).
    """
    
    if write_segments:
        all_ifos = {}

        for ifo, segment_name, version in split_segment_ids(included_segments_string.split(',')):
            all_ifos[ifo] = True


        new_seg_def_id = add_to_segment_definer(doc, proc_id, ''.join(all_ifos.keys()), 'result', 0)
        add_to_segment_summary(doc, proc_id, new_seg_def_id, [[gps_start_time, gps_end_time]])

    result = segmentlist([])

    for ifo, segment_name, version in split_segment_ids(included_segments_string.split(',')):
        sum_segments, seg_segments = build_segment_list(engine, gps_start_time, gps_end_time, ifo, segment_name, version, start_pad, end_pad)
        seg_def_id                 = add_to_segment_definer(doc, proc_id, ifo, segment_name, version)

        add_to_segment_summary(doc, proc_id, seg_def_id, sum_segments)

        # and accumulate segments 
        result |= seg_segments

    # Excluded segments are not required
    if excluded_segments_string:
        excluded_segments = segmentlist([])

        for ifo, segment_name, version in split_segment_ids(excluded_segments_string.split(',')):
            sum_segments, seg_segments = build_segment_list(engine, gps_start_time, gps_end_time, ifo, segment_name, version)
            excluded_segments |= seg_segments

        result = result - excluded_segments

    result.coalesce()
    
    # Add the segments
    if write_segments:
        add_to_segment(doc, proc_id, new_seg_def_id, result)

    return result