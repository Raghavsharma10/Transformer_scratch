def check_allowed_letters(seq, allowed_letters_as_set):
    """Validate sequence: Rise an error if sequence contains undesirable letters.  """

    # set of unique letters in sequence
    seq_set = set(seq)

    not_allowed_letters_in_seq = [x for x in seq_set if str(x).upper() not in allowed_letters_as_set]

    if len(not_allowed_letters_in_seq) > 0:
        raise forms.ValidationError(
            "This sequence type cannot contain letters: " + ", ".join(not_allowed_letters_in_seq))