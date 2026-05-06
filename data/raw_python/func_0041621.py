def _next_lexem(self, lexem_type, source_code, source_code_size):
        """Return next readable lexem of given type in source_code.
        If no value can be found, the neutral_value will be used"""
        # define reader as a lexem extractor
        def reader(seq, block_size):
            identificator = ''
            for char in source_code:
                if len(identificator) == self.idnt_values_size[lexem_type]:
                    yield self.table_values[lexem_type][identificator]
                    identificator = ''
                identificator += char
        lexem_reader = reader(source_code, self.idnt_values_size)
        lexem = None
        time_out = 0
        while lexem == None and time_out < 2*source_code_size: 
            lexem = next(lexem_reader)
            time_out += 1
        # here we have found a lexem
        return lexem