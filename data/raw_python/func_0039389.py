def _func_filters(self, filters):
        '''Build post query filters
        '''
        if not isinstance(filters, (list,tuple)):
            raise TypeError('func_filters must be a <type list> or <type tuple>')

        for i, func in enumerate(filters) :
            if isinstance(func, str) and func == 'reverse':
                filters[i] = 'reverse()'
            elif isinstance(func, tuple) and func[0] in YQL.FUNC_FILTERS:
                filters[i] = '{:s}(count={:d})'.format(*func)
            elif isinstance(func, dict) :
                func_stmt = ''
                func_name = list(func.keys())[0] # Because of Py3
                values = [ "{0}='{1}'".format(v[0], v[1]) for v in func[func_name] ]
                func_stmt = ','.join(values)
                func_stmt = '{0}({1})'.format(func_name, func_stmt)
                filters[i] = func_stmt
            else:
                raise TypeError('{0} is neither a <str>, a <tuple> or a <dict>'.format(func))
        return '| '.join(filters)