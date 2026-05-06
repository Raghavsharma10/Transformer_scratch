def t_multiline_OPTION_AND_VALUE(self, t):
        r'[^\r\n]+'
        t.lexer.multiline_newline_seen = False

        if t.value.endswith('\\'):
            return

        t.type = "OPTION_AND_VALUE"
        t.lexer.begin('INITIAL')

        value = t.lexer.lexdata[t.lexer.code_start:t.lexer.lexpos + 1]
        t.lexer.lineno += len(re.findall(r'\r\n|\n|\r', value))
        value = value.replace('\\\n', '').replace('\r', '').replace('\n', '')

        option, value = self._parse_option_value(value)

        process, option, value = self._pre_parse_value(option, value)
        if not process:
            return

        t.value = option, value

        return t