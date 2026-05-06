def transform_text(input_txt):
    """Transforms text into :py:class:`~collections.deque`, pre-processes
    multiline strings.

    :param str or bytes input_txt: Input text.
    :return: Double-ended queue of single characters and multiline strings.
    :rtype: :py:class:`~collections.deque`
    """
    if isinstance(input_txt, str):
        text = u"{}".format(input_txt)
    elif isinstance(input_txt, bytes):
        text = input_txt.decode("utf-8")
    else:
        raise TypeError("Expecting <class 'str'> or <class 'bytes'>, but {} was passed".format(type(input_txt)))

    inputq = deque(text.split(u"\n"))
    outputq = deque()

    while len(inputq) > 0:
        line = inputq.popleft()

        if line.lstrip().startswith(u"#"):
            comment = u"" + line + u"\n"
            line = inputq.popleft()

            while line.lstrip().startswith(u"#"):
                comment += line + u"\n"
                line = inputq.popleft()

            outputq.append(comment)

            for character in line:
                outputq.append(character)

        elif line.startswith(u";"):
            multiline = u""
            multiline += line + u"\n"
            line = inputq.popleft()

            while not line.startswith(u";"):
                multiline += line + u"\n"
                line = inputq.popleft()

            multiline += line[:1]
            outputq.append(multiline[1:-1])  # remove STAR syntax from multiline string

            for character in line[1:]:
                outputq.append(character)

        else:
            for character in line:
                outputq.append(character)

        outputq.append(u"\n")

    outputq.extend([u"\n", u""])  # end of file signal

    return outputq