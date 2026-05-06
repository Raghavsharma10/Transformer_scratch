def list_syntax(self):
        '''
        Prints a list of available syntax for the current paste service
        '''
        syntax_list = ['Available syntax for %s:' %(self)]
        logging.info(syntax_list[0])
        for key in self.SYNTAX_DICT.keys():
            syntax = '\t%-20s%-30s' %(key, self.SYNTAX_DICT[key])
            logging.info(syntax)
            syntax_list.append(syntax)

        return syntax_list