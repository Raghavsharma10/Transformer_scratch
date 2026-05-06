def delimit_words(string: str) -> Generator[str, None, None]:
    """
    Delimit a string at word boundaries.

    ::

        >>> import uqbar.strings
        >>> list(uqbar.strings.delimit_words("i want to believe"))
        ['i', 'want', 'to', 'believe']

    ::

        >>> list(uqbar.strings.delimit_words("S3Bucket"))
        ['S3', 'Bucket']

    ::

        >>> list(uqbar.strings.delimit_words("Route53"))
        ['Route', '53']

    """
    # TODO: Reimplement this
    wordlike_characters = ("<", ">", "!")
    current_word = ""
    for i, character in enumerate(string):
        if (
            not character.isalpha()
            and not character.isdigit()
            and character not in wordlike_characters
        ):
            if current_word:
                yield current_word
                current_word = ""
        elif not current_word:
            current_word += character
        elif character.isupper():
            if current_word[-1].isupper():
                current_word += character
            else:
                yield current_word
                current_word = character
        elif character.islower():
            if current_word[-1].isalpha():
                current_word += character
            else:
                yield current_word
                current_word = character
        elif character.isdigit():
            if current_word[-1].isdigit() or current_word[-1].isupper():
                current_word += character
            else:
                yield current_word
                current_word = character
        elif character in wordlike_characters:
            if current_word[-1] in wordlike_characters:
                current_word += character
            else:
                yield current_word
                current_word = character
    if current_word:
        yield current_word