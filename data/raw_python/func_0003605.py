def t_OPTION_AND_VALUE(self, t):
        r'[^ \n\r\t=#]+[ \t=]+[^\r\n#]+'  # TODO(etingof) escape hash
        if t.value.endswith('\\'):
            t.lexer.multiline_newline_seen = False
            t.lexer.code_start = t.lexer.lexpos - len(t.value)
            t.lexer.begin('multiline')
            return

        lineno = len(re.findall(r'\r\n|\n|\r', t.value))

        option, value = self._parse_option_value(t.value)

        process, option, value = self._pre_parse_value(option, value)
        if not process:
            return

        if value.startswith('<<'):
            t.lexer.heredoc_anchor = value[2:].strip()
            t.lexer.heredoc_option = option
            t.lexer.code_start = t.lexer.lexpos
            t.lexer.begin('heredoc')
            return

        t.value = option, value

        t.lexer.lineno += lineno

        return t