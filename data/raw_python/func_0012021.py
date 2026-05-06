def DyStrData(cls, name, regx, index = 0):
        ''' set dynamic value from the string data of response  
        @param name: glob parameter name
        @param regx: re._pattern_type
            e.g.
            DyStrData("a",re.compile('123'))
        '''
        text = Web.PageSource()
        if not text:
            return
        if not isinstance(regx, re._pattern_type):
            raise Exception("DyStrData need the arg which have compiled the regular expression.")
            
        values = regx.findall(text)
        result = ""
        if len(values)>index:
            result = values[index]        
        cls.glob.update({name:result})