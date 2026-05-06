def _value_wrapper(self, value):
        ''' wrapper for values 
        '''
        if isinstance(value, (int, float,)):
            return '=%s' % value
        elif isinstance(value, (str, unicode)):
            value = value.strip()
            # LIKE
            if RE_LIKE.match(value):
                return ' LIKE %s' % repr(RE_LIKE.match(value).group('RE_LIKE'))
            # REGEXP
            elif RE_REGEXP.match(value):
                return ' REGEXP %s' % repr(RE_REGEXP.search(value).group('RE_REGEXP'))
            else:            
                return '=%s' % repr(value)
        elif value is None:
            return ' ISNULL'