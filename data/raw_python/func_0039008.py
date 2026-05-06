def replace_refs_factory(references, use_cleveref_default, use_eqref,
                         plusname, starname, target):
    """Returns replace_refs(key, value, fmt, meta) action that replaces
    references with format-specific content.  The content is determined using
    the 'references' dict, which associates reference labels with numbers or
    string tags (e.g., { 'fig:1':1, 'fig:2':2, ...}).  If 'use_cleveref_default'
    is True, or if "modifier" in the reference's attributes is "+" or "*", then
    clever referencing is used; i.e., a name is placed in front of the number
    or string tag.  The 'plusname' and 'starname' lists give the singular
    and plural names for "+" and "*" clever references, respectively.  The
    'target' is the LaTeX type for clever referencing (e.g., "figure",
    "equation", "table", ...)."""

    global _cleveref_tex_flag  # pylint: disable=global-statement

    # Update global if clever referencing is required by default
    _cleveref_tex_flag = _cleveref_tex_flag or use_cleveref_default

    def _insert_cleveref_fakery(key, value, meta):
        r"""Inserts TeX to support clever referencing in LaTeX documents
        if the key isn't a RawBlock.  If the key is a RawBlock, then check
        the value to see if the TeX was already inserted.

        The \providecommand macro is used to fake the cleveref package's
        behaviour if it is not provided in the template via
        \usepackage{cleveref}.

        TeX is inserted into the value.  Replacement elements are returned.
        """

        global _cleveref_tex_flag  # pylint: disable=global-statement

        comment1 = '% pandoc-xnos: cleveref formatting'
        tex1 = [comment1,
                r'\crefformat{%s}{%s~#2#1#3}'%(target, plusname[0]),
                r'\Crefformat{%s}{%s~#2#1#3}'%(target, starname[0])]

        if key == 'RawBlock':  # Check for existing cleveref TeX
            if value[1].startswith(comment1):
                # Append the new portion
                value[1] = value[1] + '\n' + '\n'.join(tex1[1:])
                _cleveref_tex_flag = False  # Cleveref fakery already installed

        elif key != 'RawBlock':  # Write the cleveref TeX
            _cleveref_tex_flag = False  # Cancels further attempts
            ret = []

            # Check first to see if fakery is turned off
            if not 'xnos-cleveref-fake' in meta or \
              check_bool(get_meta(meta, 'xnos-cleveref-fake')):
                # Cleveref fakery
                tex2 = [
                    r'% pandoc-xnos: cleveref fakery',
                    r'\newcommand{\plusnamesingular}{}',
                    r'\newcommand{\starnamesingular}{}',
                    r'\newcommand{\xrefname}[1]{'\
                      r'\protect\renewcommand{\plusnamesingular}{#1}}',
                    r'\newcommand{\Xrefname}[1]{'\
                      r'\protect\renewcommand{\starnamesingular}{#1}}',
                    r'\providecommand{\cref}{\plusnamesingular~\ref}',
                    r'\providecommand{\Cref}{\starnamesingular~\ref}',
                    r'\providecommand{\crefformat}[2]{}',
                    r'\providecommand{\Crefformat}[2]{}']
                ret.append(RawBlock('tex', '\n'.join(tex2)))
            ret.append(RawBlock('tex', '\n'.join(tex1)))
            return ret
        return None

    def _cite_replacement(key, value, fmt, meta):
        """Returns context-dependent content to replace a Cite element."""

        assert key == 'Cite'

        attrs, label = value[0], _get_label(key, value)
        attrs = PandocAttributes(attrs, 'pandoc')

        assert label in references

        # Get the replacement value
        text = str(references[label])

        # Choose between \Cref, \cref and \ref
        use_cleveref = attrs['modifier'] in ['*', '+'] \
          if 'modifier' in attrs.kvs else use_cleveref_default
        plus = attrs['modifier'] == '+' if 'modifier' in attrs.kvs \
          else use_cleveref_default
        name = plusname[0] if plus else starname[0]  # Name used by cref

        # The replacement depends on the output format
        if fmt == 'latex':
            if use_cleveref:
                # Renew commands needed for cleveref fakery
                if not 'xnos-cleveref-fake' in meta or \
                  check_bool(get_meta(meta, 'xnos-cleveref-fake')):
                    faketex = (r'\xrefname' if plus else r'\Xrefname') + \
                      '{%s}' % name
                else:
                    faketex = ''
                macro = r'\cref' if plus else r'\Cref'
                ret = RawInline('tex', r'%s%s{%s}'%(faketex, macro, label))
            elif use_eqref:
                ret = RawInline('tex', r'\eqref{%s}'%label)
            else:
                ret = RawInline('tex', r'\ref{%s}'%label)
        else:
            if use_eqref:
                text = '(' + text + ')'

            linktext = [Math({"t":"InlineMath", "c":[]}, text[1:-1]) \
               if text.startswith('$') and text.endswith('$') \
               else Str(text)]

            link = elt('Link', 2)(linktext, ['#%s' % label, '']) \
              if _PANDOCVERSION < '1.16' else \
              Link(['', [], []], linktext, ['#%s' % label, ''])
            ret = ([Str(name), Space()] if use_cleveref else []) + [link]

        return ret

    def replace_refs(key, value, fmt, meta):  # pylint: disable=unused-argument
        """Replaces references with format-specific content."""

        if fmt == 'latex' and _cleveref_tex_flag:

            # Put the cleveref TeX fakery in front of the first block element
            # that isn't a RawBlock.

            if not key in ['Plain', 'Para', 'CodeBlock', 'RawBlock',
                           'BlockQuote', 'OrderedList', 'BulletList',
                           'DefinitionList', 'Header', 'HorizontalRule',
                           'Table', 'Div', 'Null']:
                return None

            # Reconstruct the block element
            el = _getel(key, value)

            # Insert cleveref TeX in front of the block element
            tex = _insert_cleveref_fakery(key, value, meta)
            if tex:
                return  tex + [el]

        elif key == 'Cite' and len(value) == 3:  # Replace the reference

            return _cite_replacement(key, value, fmt, meta)

        return None

    return replace_refs