def _parse(self, params):
        ''' parse parameters and return SQL 
        '''
        if not isinstance(params, dict):
            return None, None

        if len(params) == 0:
            return None, None

        selectors = list()
        modifiers = list()
        
        for k in params.keys():

            if k in LOGICAL_OPERATORS:
                selectors.append(self._logical(k, params[k]))

            elif k in QUERY_MODIFIERS:
                modifiers.append(self._modifier(k, params[k]))

            else:
                if k == '_id':
                    selectors.append("rowid%s" % (self._value_wrapper(params[k])))
                else:
                    selectors.append("%s%s" % (k, self._value_wrapper(params[k])))

        _selectors = ' AND '.join(selectors).strip()
        _modifiers = ' '.join(modifiers).strip()
        return _selectors, _modifiers