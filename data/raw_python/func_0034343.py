def run_file_operation(outdoc, filenames, use_segment_table, operation, preserve = True):
    """
    Performs an operation (intersect or union) across a set of files.
    That is, given a set of files each with segment definers DMT-FLAG1,
    DMT-FLAG2 etc the result is a file where 

    DMT-FLAG1 = (file 1's DMT-FLAG1 operation file 2's DMT-FLAG1 operation ...)
    DMT-FLAG2 = (file 1's DMT-FLAG2 operation file 2's DMT-FLAG2 operation ...)
    
    etc
    """

    proc_id = table.get_table(outdoc, lsctables.ProcessTable.tableName)[0].process_id

    # load up the files into individual documents
    xmldocs = [ligolw_add.ligolw_add(ligolw.Document(), [fname]) for fname in filenames]


    # Get the list of dinstinct segment_definers across all docs
    segment_definers = {}

    def register_definer(seg_def):
        key = (seg_def.ifos, seg_def.name, seg_def.version)
        segment_definers[key] = True
        return key

    for xmldoc in xmldocs:
        seg_def_table = table.get_table(xmldoc, lsctables.SegmentDefTable.tableName)
        map (register_definer, seg_def_table)

    # For each unique segment definer, find the intersection
    for ifo, name, version in segment_definers:
        if operation == INTERSECT:
            # If I were feeling especially functional-ist I'd write this
            # with reduce()
            result = pycbc_glue.segments.segmentlist([pycbc_glue.segments.segment(-pycbc_glue.segments.infinity(), pycbc_glue.segments.infinity())])
            for xmldoc in xmldocs:
                result &= find_segments(xmldoc, '%s:%s:%d' % (ifo, name, version), use_segment_table)
        elif operation == UNION:
            result = pycbc_glue.segments.segmentlist([])

            for xmldoc in xmldocs:
                result |= find_segments(xmldoc, '%s:%s:%d' % (ifo, name, version), use_segment_table)
        elif operation == DIFF:
            result = find_segments(xmldocs[0], '%s:%s:%d' % (ifo, name, version), use_segment_table)
            
            for xmldoc in xmldocs[1:]:
                result -= find_segments(xmldoc, '%s:%s:%d' % (ifo, name, version), use_segment_table)
        else:
            raise NameError ("%s is not a known operation (intersect, union or diff)" % operation)


        # Add a segment definer for the result
        seg_def_id = add_to_segment_definer(outdoc, proc_id, ifo, name, version)

        # Add the segments
        if use_segment_table:
            add_to_segment(outdoc, proc_id, seg_def_id, result)
        else:
            add_to_segment_summary(outdoc, proc_id, seg_def_id, result)

    # If we're preserving, also load up everything into the output document.
    if preserve:
        # Add them to the output document
        map(lambda x: outdoc.appendChild(x.childNodes[0]), xmldocs)

        # Merge the ligolw elements and tables
        ligolw_add.merge_ligolws(outdoc)
        ligolw_add.merge_compatible_tables(outdoc)

    return outdoc, abs(result)