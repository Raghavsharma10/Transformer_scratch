def _logical(self, operator, params):
        ''' 
        $and:   joins query clauses with a logical AND returns all items 
                that match the conditions of both clauses
        $or:    joins query clauses with a logical OR returns all items 
                that match the conditions of either clause.
        '''

        result = list()
        if isinstance(params, dict):
            for k,v in params.items():
                selectors, modifiers = self._parse(dict([(k, v),]))
                result.append("(%s)" % selectors) 
        elif isinstance(params, (list, tuple)):
            for v in params:
                selectors, modifiers = self._parse(v)
                result.append("(%s)" % selectors)
        else:
            raise RuntimeError('Unknow parameter type, %s:%s' % (type(params), params))

        if operator == '$and':
            return ' AND '.join(result)
        elif operator == '$or':
            return ' OR '.join(result)
        else:
            raise RuntimeError('Unknown operator, %s' % operator)