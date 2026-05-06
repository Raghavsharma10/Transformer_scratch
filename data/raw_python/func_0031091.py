def iter(self):
        """
        Iterate over the sequences in self.file_, yielding each as an
        instance of the desired read class.

        @raise ValueError: If the input file has an odd number of records or
            if any sequence has a different length than its predicted
            secondary structure.
        """
        upperCase = self._upperCase
        for _file in self._files:
            with asHandle(_file) as fp:
                records = SeqIO.parse(fp, 'fasta')
                while True:
                    try:
                        record = next(records)
                    except StopIteration:
                        break
                    try:
                        structureRecord = next(records)
                    except StopIteration:
                        raise ValueError('Structure file %r has an odd number '
                                         'of records.' % _file)

                    if len(structureRecord) != len(record):
                        raise ValueError(
                            'Sequence %r length (%d) is not equal to '
                            'structure %r length (%d) in input file %r.' % (
                                record.description, len(record),
                                structureRecord.description,
                                len(structureRecord), _file))

                    if upperCase:
                        read = self._readClass(
                            record.description,
                            str(record.seq.upper()),
                            str(structureRecord.seq.upper()))
                    else:
                        read = self._readClass(record.description,
                                               str(record.seq),
                                               str(structureRecord.seq))

                    yield read