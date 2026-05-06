def camelCaseToDashName(camelCase):
        '''
            camelCaseToDashName - Convert a camel case name to a dash-name (like paddingTop to padding-top)

            @param camelCase <str> - A camel-case string

            @return <str> - A dash-name
        '''

        camelCaseList = list(camelCase)

        ret = []

        for ch in camelCaseList:
            if ch.isupper():
                ret.append('-')
                ret.append(ch.lower())
            else:
                ret.append(ch)

        return ''.join(ret)