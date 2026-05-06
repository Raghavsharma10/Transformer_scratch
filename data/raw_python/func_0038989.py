def validate_sequence(sequence: str, sequence_is_as_nucleotide=True):
    """Validate sequence in blast/tblastn form.  """

    tmp_seq = tempfile.NamedTemporaryFile(mode="wb+", delete=False)

    if len(str(sequence).strip()) == 0:
        raise forms.ValidationError(blast_settings.BLAST_CORRECT_SEQ_ERROR_MSG)

    if str(sequence).strip()[0] != ">":
        tmp_seq.write(">seq1\n".encode())

    tmp_seq.write(sequence.encode())
    tmp_seq.close()

    records = SeqIO.index(tmp_seq.name, "fasta")
    record_count = len(records)

    if record_count == 0:
        raise forms.ValidationError(blast_settings.BLAST_CORRECT_SEQ_ERROR_MSG)

    if record_count > blast_settings.BLAST_MAX_NUMBER_SEQ_IN_INPUT:
        raise forms.ValidationError(blast_settings.BLAST_CORRECT_SEQ_MAX_SEQ_NUMB_ERROR_MSG)

    # read query sequence from temporary file
    first_sequence_list_in_file = SeqIO.parse(tmp_seq.name, "fasta")

    for sequence in first_sequence_list_in_file:

        if len(sequence.seq) <= 10:
            raise forms.ValidationError(blast_settings.BLAST_CORRECT_SEQ_TOO_SHORT_ERROR_MSG)

        if sequence_is_as_nucleotide:
            check_allowed_letters(str(sequence.seq), ALLOWED_NUCL)
        else:
            check_allowed_letters(str(sequence.seq), ALLOWED_AMINOACIDS)

    return tmp_seq