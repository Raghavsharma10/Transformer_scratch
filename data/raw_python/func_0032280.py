def select_char_code_table(self, table):
        '''Select character code table, from tree built in ones.
        
        Args:
            table: The desired character code table. Choose from 'standard', 'eastern european', 'western european', and 'spare'
        Returns:
            None
        Raises:
            RuntimeError: Invalid chartable.
        '''
        tables = {'standard': 0,
                  'eastern european': 1,
                  'western european': 2,
                  'spare': 3
                  }
        if table in tables:
            self.send(chr(27)+'t'+chr(tables[table]))
        else:
            raise RuntimeError('Invalid char table.')