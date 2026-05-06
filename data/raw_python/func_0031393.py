def _modifier(self, operator, params):
        ''' 
        $orderby:   sorts the results of a query in ascending (1) or descending (-1) order.
        '''

        if operator == '$orderby':
            order_types = {-1: 'DESC', 1: 'ASC'}
            if not isinstance(params, dict):
                raise RuntimeError('Incorrect parameter type, %s' % params) 
            return 'ORDER BY %s' % ','.join(["%s %s" % (p, order_types[params[p]]) for p in params])
        else:
            raise RuntimeError('Unknown operator, %s' % operator)