def _clause_formatter(self, cond):
        '''Formats conditions
        args is a list of ['field', 'operator', 'value']
        '''

        if len(cond) == 2 :
            cond = ' '.join(cond)
            return cond

        if 'in' in cond[1].lower() :
            if not isinstance(cond[2], (tuple, list)):
                    raise TypeError('("{0}") must be of type <type tuple> or <type list>'.format(cond[2]))

            if 'select' not in cond[2][0].lower() :
                cond[2] = "({0})".format(','.join(map(str,["'{0}'".format(e) for e in cond[2]])))
            else:
                cond[2] = "({0})".format(','.join(map(str,["{0}".format(e) for e in cond[2]])))

            cond = " ".join(cond)
        else: 
            #if isinstance(cond[2], str):
            #    var = re.match('^@(\w+)$', cond[2])
            #else:
            #    var = None
            #if var :
            if isinstance(cond[2], str) and cond[2].startswith('@'):
                cond[2] = "{0}".format(cond[2])
            else :
                cond[2] = "'{0}'".format(cond[2])
            cond = ' '.join(cond)

        return cond