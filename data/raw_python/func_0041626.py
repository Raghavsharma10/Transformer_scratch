def _create_values_table(self):
        """Create table lexem_type->{identificator->vocabulary}, 
        and return it with sizes of an identificator as lexem_type->identificator_size"""
        # number of existing character, and returned dicts
        len_alph = len(self.alphabet)
        identificators_table = {k:{} for k in self.voc_values.keys()}
        identificators_sizes = {k:-1 for k in self.voc_values.keys()}

        for lexem_type, vocabulary in self.voc_values.items():
            # find number of different values that can be found, 
            #   and size of an identificator.
            len_vocb = len(vocabulary)
            identificators_sizes[lexem_type] = ceil(log(len_vocb, len_alph))
            # create list of possible identificators 
            num2alph = lambda x, n: self.alphabet[(x // len_alph**n) % len_alph]
            identificators = [[str(num2alph(x, n)) 
                               for n in range(identificators_sizes[lexem_type])
                              ] # this list is an identificator
                              for x in range(len_alph**identificators_sizes[lexem_type])
                             ] # this one is a list of identificator
            # initialize iterable
            zip_id_voc = zip_longest(
                identificators, vocabulary, 
                fillvalue=None
            )
            # create dict {identificator:word}
            for idt, voc in zip_id_voc:
                identificators_table[lexem_type][''.join(idt)] = voc

        # return all 
        return identificators_table, identificators_sizes