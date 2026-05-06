def collapse(
    function,
    open_string,
    close_string,
    replacement='',
    exceptions=None,
):
    """Collapses the text between two delimiters in a frame function value

    This collapses the text between two delimiters and either removes the text
    altogether or replaces it with a replacement string.

    There are certain contexts in which we might not want to collapse the text
    between two delimiters. These are denoted as "exceptions" and collapse will
    check for those exception strings occuring before the token to be replaced
    or inside the token to be replaced.

    Before::

        IPC::ParamTraits<nsTSubstring<char> >::Write(IPC::Message *,nsTSubstring<char> const &)
               ^        ^ open token
               exception string occurring before open token

    Inside::

        <rayon_core::job::HeapJob<BODY> as rayon_core::job::Job>::execute
        ^                              ^^^^ exception string inside token
        open token

    :arg function: the function value from a frame to collapse tokens in
    :arg open_string: the open delimiter; e.g. ``(``
    :arg close_string: the close delimiter; e.g. ``)``
    :arg replacement: what to replace the token with; e.g. ``<T>``
    :arg exceptions: list of strings denoting exceptions where we don't want
        to collapse the token

    :returns: new function string with tokens collapsed

    """
    collapsed = []
    open_count = 0
    open_token = []

    for i, char in enumerate(function):
        if not open_count:
            if char == open_string and not _is_exception(exceptions, function[:i], function[i + 1:], ''):  # noqa
                open_count += 1
                open_token = [char]
            else:
                collapsed.append(char)

        else:
            if char == open_string:
                open_count += 1
                open_token.append(char)

            elif char == close_string:
                open_count -= 1
                open_token.append(char)

                if open_count == 0:
                    token = ''.join(open_token)
                    if _is_exception(exceptions, function[:i], function[i + 1:], token):
                        collapsed.append(''.join(open_token))
                    else:
                        collapsed.append(replacement)
                    open_token = []
            else:
                open_token.append(char)

    if open_count:
        token = ''.join(open_token)
        if _is_exception(exceptions, function[:i], function[i + 1:], token):
            collapsed.append(''.join(open_token))
        else:
            collapsed.append(replacement)

    return ''.join(collapsed)