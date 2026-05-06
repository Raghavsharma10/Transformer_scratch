def parse_journal_file(journal_file):
    """Iterates over the journal's file taking care of paddings."""
    counter = count()

    for block in read_next_block(journal_file):
        block = remove_nullchars(block)

        while len(block) > MIN_RECORD_SIZE:
            header = RECORD_HEADER.unpack_from(block)
            size = header[0]

            try:
                yield parse_record(header, block[:size])

                next(counter)
            except RuntimeError:
                yield CorruptedUsnRecord(next(counter))
            finally:
                block = remove_nullchars(block[size:])

        journal_file.seek(- len(block), 1)