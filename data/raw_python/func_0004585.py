def rbnf_lexing(text: str):
    """Read loudly for documentation."""

    cast_map: const    = _cast_map
    lexer_table: const = _lexer_table
    keyword: const     = _keyword
    drop_table: const  = _DropTable
    end: const         = _END
    unknown: const     = _UNKNOWN

    text_length = len(text)
    colno       = 0
    lineno      = 0
    position    = 0

    cast_const  = ConstStrPool.cast_to_const

    while True:
        if text_length <= position:
            break

        for case_name, text_match_case in lexer_table:
            matched_text = text_match_case(text, position)
            if not matched_text:
                continue

            case_mem_addr = id(case_name)  # memory address of case_name

            if case_mem_addr not in drop_table:

                if matched_text in cast_map:
                    yield Tokenizer(keyword, cast_const(matched_text), lineno, colno)

                else:
                    yield Tokenizer(cast_const(case_name), matched_text, lineno, colno)

            n = len(matched_text)
            line_inc = matched_text.count('\n')

            if line_inc:

                latest_newline_idx = matched_text.rindex('\n')
                colno = n - latest_newline_idx
                lineno += line_inc

                if case_name is _Space and matched_text[-1] == '\n':

                    yield Tokenizer(end, '', lineno, colno)

            else:
                colno += n
            position += n
            break

        else:

            char = text[position]
            yield Tokenizer(unknown, char, lineno, colno)

            position += 1

            if char == '\n':
                lineno += 1
                colno = 0

            else:
                colno += 1