def _convertBlastRecordToDict(self, record):
        """
        Pull (only) the fields we use out of the record and return them as a
        dict.  Although we take the title from each alignment description, we
        save space in the JSON output by storing it in the alignment dict (not
        in a separated 'description' dict). When we undo this conversion (in
        JSONRecordsReader._convertDictToBlastRecord) we'll pull the title out
        of the alignment dict and put it into the right place in the BLAST
        record.

        @param record: An instance of C{Bio.Blast.Record.Blast}. The attributes
            on this don't seem to be documented. You'll need to look at the
            BioPython source to see everything it contains.
        @return: A C{dict} with 'alignments' and 'query' keys.
        """
        alignments = []
        for alignment in record.alignments:
            hsps = []
            for hsp in alignment.hsps:
                hsps.append({
                    'bits': hsp.bits,
                    'expect': hsp.expect,
                    'frame': hsp.frame,
                    'identicalCount': hsp.identities,
                    'positiveCount': hsp.positives,
                    'query': hsp.query,
                    'query_start': hsp.query_start,
                    'query_end': hsp.query_end,
                    'sbjct': hsp.sbjct,
                    'sbjct_start': hsp.sbjct_start,
                    'sbjct_end': hsp.sbjct_end,
                })

            alignments.append({
                'hsps': hsps,
                'length': alignment.length,
                'title': alignment.title,
            })

        return {
            'alignments': alignments,
            'query': record.query,
        }