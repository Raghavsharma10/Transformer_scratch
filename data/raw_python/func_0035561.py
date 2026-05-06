def ReadCodonAlignment(fastafile, checknewickvalid):
    """Reads codon alignment from file.

    *fastafile* is the name of an existing FASTA file.

    *checknewickvalid* : if *True*, we require that names are unique and do
    **not** contain spaces, commas, colons, semicolons, parentheses, square
    brackets, or single or double quotation marks.
    If any of these disallowed characters are present, raises an Exception.

    Reads the alignment from the *fastafile* and returns the aligned
    sequences as a list of 2-tuple of strings *(header, sequence)*
    where *sequence* is upper case.

    If the terminal codon is a stop codon for **all** sequences, then
    this terminal codon is trimmed. Raises an exception if the sequences
    are not aligned codon sequences that are free of stop codons (with
    the exception of a shared terminal stop) and free of ambiguous nucleotides.

    Read aligned sequences in this example:

    >>> seqs = [('seq1', 'ATGGAA'), ('seq2', 'ATGAAA')]
    >>> f = io.StringIO()
    >>> n = f.write(u'\\n'.join(['>{0}\\n{1}'.format(*tup) for tup in seqs]))
    >>> n = f.seek(0)
    >>> a = ReadCodonAlignment(f, True)
    >>> seqs == a
    True

    Trim stop codons from all sequences in this example:

    >>> seqs = [('seq1', 'ATGTAA'), ('seq2', 'ATGTGA')]
    >>> f = io.StringIO()
    >>> n = f.write(u'\\n'.join(['>{0}\\n{1}'.format(*tup) for tup in seqs]))
    >>> n = f.seek(0)
    >>> a = ReadCodonAlignment(f, True)
    >>> [(head, seq[ : -3]) for (head, seq) in seqs] == a
    True

    Read sequences with gap:

    >>> seqs = [('seq1', 'ATG---'), ('seq2', 'ATGAGA')]
    >>> f = io.StringIO()
    >>> n = f.write(u'\\n'.join(['>{0}\\n{1}'.format(*tup) for tup in seqs]))
    >>> n = f.seek(0)
    >>> a = ReadCodonAlignment(f, True)
    >>> [(head, seq) for (head, seq) in seqs] == a
    True

    Premature stop codon gives error:

    >>> seqs = [('seq1', 'TGAATG'), ('seq2', 'ATGAGA')]
    >>> f = io.StringIO()
    >>> n = f.write(u'\\n'.join(['>{0}\\n{1}'.format(*tup) for tup in seqs]))
    >>> n = f.seek(0)
    >>> a = ReadCodonAlignment(f, True) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:
    """
    codonmatch = re.compile('^[ATCG]{3}$')
    gapmatch = re.compile('^-+^')
    seqs = [(seq.description.strip(), str(seq.seq).upper()) for seq
            in Bio.SeqIO.parse(fastafile, 'fasta')]
    assert seqs, "{0} failed to specify any sequences".format(fastafile)

    seqlen = len(seqs[0][1])
    if not all([len(seq) == seqlen for (head, seq) in seqs]):
        raise ValueError(("All sequences in {0} are not of the same length; "
                "they must not be properly aligned").format(fastafile))
    if (seqlen < 3) or (seqlen % 3 != 0):
        raise ValueError(("The length of the sequences in {0} is {1} which "
                "is not divisible by 3; they are not valid codon sequences"
                ).format(fastafile, seqlen))

    terminalcodon = []
    codons_by_position = dict([(icodon, []) for icodon in range(seqlen // 3)])
    for (head, seq) in seqs:
        assert len(seq) % 3 == 0
        for icodon in range(seqlen // 3):
            codon = seq[3 * icodon : 3 * icodon + 3]
            codons_by_position[icodon].append(codon)
            if codonmatch.search(codon):
                aa = str(Bio.Seq.Seq(codon).translate())
                if aa == '*':
                    if icodon + 1 != len(seq) // 3:
                        raise ValueError(("In {0}, sequence {1}, non-terminal "
                                "codon {2} is stop codon: {3}").format(
                                fastafile, head, icodon + 1, codon))
            elif codon == '---':
                aa = '-'
            else:
                raise ValueError(("In {0}, sequence {1}, codon {2} is invalid: "
                        "{3}").format(fastafile, head, icodon + 1, codon))
        terminalcodon.append(aa)

    for (icodon, codonlist) in codons_by_position.items():
        if all([codon == '---' for codon in codonlist]):
            raise ValueError(("In {0}, all codons are gaps at position {1}"
                    ).format(fastafile, icodon + 1))

    if all([aa in ['*', '-'] for aa in terminalcodon]):
        if len(seq) == 3:
            raise ValueError(("The only codon is a terminal stop codon for "
                    "the sequences in {0}").format(fastafile))
        seqs = [(head, seq[ : -3]) for (head, seq) in seqs]
    elif any([aa == '*' for aa in terminalcodon]):
        raise ValueError(("Only some sequences in {0} have a terminal stop "
                "codon. All or none must have terminal stop.").format(fastafile))

    if any([gapmatch.search(seq) for (head, seq) in seqs]):
        raise ValueError(("In {0}, at least one sequence is entirely composed "
                "of gaps.").format(fastafile))

    if checknewickvalid:
        if len(set([head for (head, seq) in seqs])) != len(seqs):
            raise ValueError("Headers in {0} not all unique".format(fastafile))
        disallowedheader = re.compile('[\s\:\;\(\)\[\]\,\'\"]')
        for (head, seq) in seqs:
            if disallowedheader.search(head):
                raise ValueError(("Invalid character in header in {0}:"
                        "\n{2}").format(fastafile, head))

    return seqs