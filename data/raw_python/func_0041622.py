def _next_condition_lexems(self, source_code, source_code_size):
        """Return condition lexem readed in source_code"""
        # find three lexems
        lexems = tuple((
            self._next_lexem(LEXEM_TYPE_COMPARISON, source_code, source_code_size),
            self._next_lexem(LEXEM_TYPE_OPERATOR  , source_code, source_code_size),
            self._next_lexem(LEXEM_TYPE_COMPARISON, source_code, source_code_size)
        ))
        # verify integrity
        if None in lexems: # one of the condition lexem was not found in source code 
            return None
        else: # all lexems are valid
            return ' '.join(lexems)