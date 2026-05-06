def addFile(self, filename):
        """
        Add a new FASTA file of sequences.

        @param filename: A C{str} file name, with the file in FASTA format.
            This file must (obviously) exist at indexing time. When __getitem__
            is used to access sequences, it is possible to provide a
            C{fastaDirectory} argument to our C{__init__} to indicate the
            directory containing the original FASTA files, in which case the
            basename of the file here provided in C{filename} is used to find
            the file in the given directory. This allows the construction of a
            sqlite database from the shell in one directory and its use
            programmatically from another directory.
        @raise ValueError: If a file with this name has already been added or
            if the file contains a sequence whose id has already been seen.
        @return: The C{int} number of sequences added from the file.
        """
        endswith = filename.lower().endswith
        if endswith('.bgz') or endswith('.gz'):
            useBgzf = True
        elif endswith('.bz2'):
            raise ValueError(
                'Compressed FASTA is only supported in BGZF format. Use '
                'bgzip to compresss your FASTA.')
        else:
            useBgzf = False

        fileNumber = self._addFilename(filename)
        connection = self._connection
        count = 0
        try:
            with connection:
                if useBgzf:
                    try:
                        fp = bgzf.open(filename, 'rb')
                    except ValueError as e:
                        if str(e).find('BGZF') > -1:
                            raise ValueError(
                                'Compressed FASTA is only supported in BGZF '
                                'format. Use the samtools bgzip utility '
                                '(instead of gzip) to compresss your FASTA.')
                        else:
                            raise
                    else:
                        try:
                            for line in fp:
                                if line[0] == '>':
                                    count += 1
                                    id_ = line[1:].rstrip(' \t\n\r')
                                    connection.execute(
                                        'INSERT INTO sequences(id, '
                                        'fileNumber, offset) VALUES (?, ?, ?)',
                                        (id_, fileNumber, fp.tell()))
                        finally:
                            fp.close()
                else:
                    with open(filename) as fp:
                        offset = 0
                        for line in fp:
                            offset += len(line)
                            if line[0] == '>':
                                count += 1
                                id_ = line[1:].rstrip(' \t\n\r')
                                connection.execute(
                                    'INSERT INTO sequences(id, fileNumber, '
                                    'offset) VALUES (?, ?, ?)',
                                    (id_, fileNumber, offset))
        except sqlite3.IntegrityError as e:
            if str(e).find('UNIQUE constraint failed') > -1:
                original = self._find(id_)
                if original is None:
                    # The id must have appeared twice in the current file,
                    # because we could not look it up in the database
                    # (i.e., it was INSERTed but not committed).
                    raise ValueError(
                        "FASTA sequence id '%s' found twice in file '%s'." %
                        (id_, filename))
                else:
                    origFilename, _ = original
                    raise ValueError(
                        "FASTA sequence id '%s', found in file '%s', was "
                        "previously added from file '%s'." %
                        (id_, filename, origFilename))
            else:
                raise
        else:
            return count