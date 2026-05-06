def run_segment_operation(outdoc, filenames, segments, use_segment_table, operation, result_name = 'RESULT', preserve = True):
    """
    Performs an operation (intersect or union) across a set of segments.
    That is, given a set of files each with segment definers DMT-FLAG1,
    DMT-FLAG2 etc and a list of segments DMT-FLAG1,DMT-FLAG1 this returns

    RESULT = (table 1's DMT-FLAG1 union table 2's DMT-FLAG1 union ...)
             operation
             (table 1's DMT-FLAG2 union table 2's DMT-FLAG2 union ...)
             operation
    etc
    """

    proc_id = table.get_table(outdoc, lsctables.ProcessTable.tableName)[0].process_id

    if preserve:
        indoc = ligolw_add.ligolw_add(outdoc, filenames)
    else:
        indoc = ligolw_add.ligolw_add(ligolw.Document(), filenames)

    # Start with a segment covering all of time, then
    # intersect with each of the fields of interest
    keys = segments.split(',')

    if operation == INTERSECT:
        sgmntlist = pycbc_glue.segments.segmentlist([pycbc_glue.segments.segment(-pycbc_glue.segments.infinity(), pycbc_glue.segments.infinity())])

        for key in keys:
            sgmntlist &= find_segments(indoc, key, use_segment_table)

    elif operation == UNION:
        sgmntlist = pycbc_glue.segments.segmentlist([])

        for key in keys:
            sgmntlist |= find_segments(indoc, key, use_segment_table)
    elif operation == DIFF:
        sgmntlist = find_segments(indoc, keys[0], use_segment_table)

        for key in keys[1:]:
            sgmntlist -= find_segments(indoc, key, use_segment_table)
    else:
        raise NameError("%s is not a known operation (intersect, union or diff)" % operation)


    # Add a segment definer and segments
    seg_def_id = add_to_segment_definer(outdoc, proc_id, '', result_name, 1)

    if use_segment_table:
        add_to_segment(outdoc, proc_id, seg_def_id, sgmntlist)
    else:
        add_to_segment_summary(outdoc, proc_id, seg_def_id, sgmntlist)

    return outdoc, abs(sgmntlist)