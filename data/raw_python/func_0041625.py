def _create_struct_table(self):
        """Create table identificator->vocabulary, 
        and return it with size of an identificator"""
        len_alph = len(self.alphabet)
        len_vocb = len(self.voc_structure)
        identificator_size = ceil(log(len_vocb, len_alph))
        # create list of lexems 
        num2alph = lambda x, n: self.alphabet[(x // len_alph**n) % len_alph]
        identificators = [[str(num2alph(x, n)) 
                           for n in range(identificator_size)
                          ] 
                          for x in range(len_vocb)
                         ]
        # initialize table and iterable
        identificators_table = {}
        zip_id_voc = zip_longest(
            identificators, self.voc_structure, 
            fillvalue=None
        )
        # create dict identificator:word
        for idt, word in zip_id_voc:
            identificators_table[''.join(idt)] = word
        return identificators_table, identificator_size