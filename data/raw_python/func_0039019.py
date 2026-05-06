def blast_records_to_object(blast_records):
    """Transforms biopython's blast record into blast object defined in django-blastplus app.  """

    # container for transformed objects
    blast_objects_list = []

    for blast_record in blast_records:

        br = BlastRecord(**{'query': blast_record.query,
                            'version': blast_record.version,
                            'expect': blast_record.expect,
                            'application': blast_record.application,
                            'reference': blast_record.reference})

        for alignment in blast_record.alignments:

            al = Alignment(**{
                'hit_def': alignment.hit_def,
                'title': alignment.title,
                'length': alignment.length,
            })

            for hsp in alignment.hsps:
                h = Hsp(**{
                    'align_length': hsp.align_length,
                    'bits': hsp.bits,
                    'expect': hsp.expect,
                    'frame': hsp.frame,
                    'gaps': hsp.gaps,
                    'identities': hsp.identities,
                    'match': hsp.match,
                    'num_alignments': hsp.num_alignments,
                    'positives': hsp.positives,
                    'query': hsp.query,
                    'query_end': hsp.query_end,
                    'query_start': hsp.query_start,
                    'sbjct': hsp.sbjct,
                    'sbjct_end': hsp.sbjct_end,
                    'sbjct_start': hsp.sbjct_start,
                    'score': hsp.score,
                    'strand': hsp.strand,
                    'str': str(hsp),
                })

                al.hsp_list.append(h)
            br.alignments.append(al)
        blast_objects_list.append(br)
    return blast_objects_list