def validate_word_size(word_size, BLAST_SETS):
    """Validate word size in blast/tblastn form.  """

    blast_min_int_word_size = BLAST_SETS.min_word_size
    blast_max_int_word_size = BLAST_SETS.max_word_size
    blast_word_size_error = BLAST_SETS.get_word_size_error()

    try:
        if len(word_size) <= 0:
            raise forms.ValidationError(blast_word_size_error)

        int_word_size = int(word_size)

        if int_word_size < blast_min_int_word_size:
            raise forms.ValidationError(blast_word_size_error)

        if int_word_size >= blast_max_int_word_size:
            raise forms.ValidationError(blast_word_size_error)

    except:
        raise forms.ValidationError(blast_word_size_error)

    return int_word_size