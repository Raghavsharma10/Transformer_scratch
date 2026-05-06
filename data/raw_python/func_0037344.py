def bmrblex(text):
    """A lexical analyzer for the BMRB NMR-STAR format syntax.

    :param text: Input text.
    :type text: :py:class:`str` or :py:class:`bytes`
    :return: Current token.
    :rtype: :py:class:`str`
    """
    stream = transform_text(text)

    wordchars = (u"abcdfeghijklmnopqrstuvwxyz"
                 u"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                 u"ßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
                 u"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"
                 u"!@$%^&*()_+:;?/>.<,~`|\{[}]-=")

    whitespace = u" \t\v\r\n"
    comment = u"#"
    state = u" "
    token = u""
    single_line_comment = u""

    while len(stream) > 0:
        nextnextchar = stream.popleft()

        while True:
            nextchar = nextnextchar

            if len(stream) > 0:
                nextnextchar = stream.popleft()
            else:
                nextnextchar = u""

            # Process multiline string, comment, or single line comment
            if len(nextchar) > 1:
                state = u" "
                token = nextchar
                break  # emit current token

            elif nextchar in whitespace and nextnextchar in comment and state not in (u"'", u'"'):
                single_line_comment = u""
                state = u"#"

            if state is None:
                token = u""  # past end of file
                break

            elif state == u" ":
                if not nextchar:
                    state = None
                    break

                elif nextchar in whitespace:
                    if token:
                        state = u" "
                        break  # emit current token
                    else:
                        continue

                elif nextchar in wordchars:
                    token = nextchar
                    state = u"a"

                elif nextchar == u"'" or nextchar == u'"':
                    token = nextchar
                    state = nextchar

                else:
                    token = nextchar
                    if token:
                        state = u" "
                        break  # emit current token
                    else:
                        continue

            # Process single-quoted or double-quoted token
            elif state == u"'" or state == u'"':
                token += nextchar
                if nextchar == state:
                    if nextnextchar in whitespace:
                        state = u" "
                        token = token[1:-1]  # remove single or double quotes from the ends
                        break

            # Process single line comment
            elif state == u"#":
                single_line_comment += nextchar
                if nextchar == u"\n":
                    state = u" "
                    break

            # Process regular (unquoted) token
            elif state == u"a":
                if not nextchar:
                    state = None
                    break
                elif nextchar in whitespace:
                    state = u" "
                    if token:
                        break  # emit current token
                    else:
                        continue
                else:
                    token += nextchar

        if nextnextchar:
            stream.appendleft(nextnextchar)

        yield token
        token = u""