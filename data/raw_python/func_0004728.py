def render(txt):
    """
    Accepts Slack formatted text and returns HTML.
    """

    # Removing links to other channels
    txt = re.sub(r'<#[^\|]*\|(.*)>', r'#\g<1>', txt)

    # Removing links to other users
    txt = re.sub(r'<(@.*)>', r'\g<1>', txt)

    # handle named hyperlinks
    txt = re.sub(r'<([^\|]*)\|([^\|]*)>', r'<a href="\g<1>" target="blank">\g<2></a>', txt)

    # handle unnamed hyperlinks
    txt = re.sub(r'<([^a|/a].*)>', r'<a href="\g<1>" target="blank">\g<1></a>', txt)

    # handle ordered and unordered lists
    for delimeter in LIST_DELIMITERS:
        slack_tag = delimeter
        class_name = LIST_DELIMITERS[delimeter]

        # Wrap any lines that start with the slack_tag in <li></li>
        list_regex = u'(?:^|\n){}\s?(.*)'.format(slack_tag)
        list_repl = r'<li class="list-item-{}">\g<1></li>'.format(class_name)
        txt = re.sub(list_regex, list_repl, txt)

    # hanlde blockquotes
    txt = re.sub(u'(^|\n)(?:&gt;){3}\s?(.*)$', r'\g<1><blockquote>\g<2></blockquote>', txt, flags=re.DOTALL)
    txt = re.sub(u'(?:^|\n)&gt;\s?(.*)\n?', r'<blockquote>\g<1></blockquote>', txt)

    # handle code blocks
    txt = re.sub(r'```\n?(.*)```', r'<pre>\g<1></pre>', txt, flags=re.DOTALL)
    txt = re.sub(r'\n(</pre>)', r'\g<1>', txt)

    # handle bolding, italics, and strikethrough
    for wrapper in FORMATTERS:
        slack_tag = wrapper
        html_tag = FORMATTERS[wrapper]

        # Grab all text in formatted characters on the same line unless escaped
        regex = r'(?<!\\)\{t}([^\{t}|\n]*)\{t}'.format(t=slack_tag)
        repl = r'<{t}>\g<1></{t}>'.format(t=html_tag)
        txt = re.sub(regex, repl, txt)

    # convert line breaks
    txt = txt.replace('\n', '<br />')

    # clean up bad HTML
    parser = CustomSlackdownHTMLParser(txt)
    txt = parser.clean()

    # convert multiple spaces
    txt = txt.replace(r'  ', ' &nbsp')

    return txt