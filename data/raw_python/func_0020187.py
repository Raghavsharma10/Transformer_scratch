def compiled(template, delimiters=DEFAULT_DELIMITERS):
    """Compile a template into token tree

    :template: TODO
    :delimiters: TODO
    :returns: the root token

    """
    re_tag = delimiters_to_re(delimiters)

    # variable to save states
    tokens = []
    index = 0
    sections = []
    tokens_stack = []

    # root token
    root = Root('root')
    root.filters = copy.copy(filters)

    m = re_tag.search(template, index)
    while m is not None:
        token = None
        last_literal = None
        strip_space = False

        if m.start() > index:
            last_literal = Literal('str', template[index:m.start()], root=root)
            tokens.append(last_literal)

        # parse token
        prefix, name, suffix = m.groups()

        if prefix == '=' and suffix == '=':
            # {{=| |=}} to change delimiters
            delimiters = re.split(r'\s+', name)
            if len(delimiters) != 2:
                raise SyntaxError('Invalid new delimiter definition: ' + m.group())
            re_tag = delimiters_to_re(delimiters)
            strip_space = True

        elif prefix == '{' and suffix == '}':
            # {{{ variable }}}
            token = Variable(name, name, root=root)

        elif prefix == '' and suffix == '':
            # {{ name }}
            token = Variable(name, name, root=root)
            token.escape = True

        elif suffix != '' and suffix != None:
            raise SyntaxError('Invalid token: ' + m.group())

        elif prefix == '&':
            # {{& escaped variable }}
            token = Variable(name, name, root=root)

        elif prefix == '!':
            # {{! comment }}
            token = Comment(name, root=root)
            if len(sections) <= 0:
                # considered as standalone only outside sections
                strip_space = True

        elif prefix == '>':
            # {{> partial}}
            token = Partial(name, name, root=root)
            strip_space = True

            pos = is_standalone(template, m.start(), m.end())
            if pos:
                token.indent = len(template[pos[0]:m.start()])

        elif prefix == '#' or prefix == '^':
            # {{# section }} or # {{^ inverted }}

            # strip filter
            sec_name = name.split('|')[0].strip()
            token = Section(sec_name, name, root=root) if prefix == '#' else Inverted(name, name, root=root)
            token.delimiter = delimiters
            tokens.append(token)

            # save the tokens onto stack
            token = None
            tokens_stack.append(tokens)
            tokens = []

            sections.append((sec_name, prefix, m.end()))
            strip_space = True

        elif prefix == '/':
            tag_name, sec_type, text_end = sections.pop()
            if tag_name != name:
                raise SyntaxError("unclosed tag: '" + tag_name + "' Got:" + m.group())

            children = tokens
            tokens = tokens_stack.pop()

            tokens[-1].text = template[text_end:m.start()]
            tokens[-1].children = children
            strip_space = True

        else:
            raise SyntaxError('Unknown tag: ' + m.group())

        if token is not None:
            tokens.append(token)

        index = m.end()
        if strip_space:
            pos = is_standalone(template, m.start(), m.end())
            if pos:
                index = pos[1]
                if last_literal: last_literal.value = last_literal.value.rstrip(spaces_not_newline)

        m = re_tag.search(template, index)

    tokens.append(Literal('str', template[index:]))
    root.children = tokens
    return root