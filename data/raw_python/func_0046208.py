def return_val(self):
        """
        Returns the return value of the function, as a ParamDoc with an
        empty name:

        >>> comments = parse_comments_for_file('examples/module_closure.js')
        >>> fn1 = FunctionDoc(comments[1])
        >>> fn1.return_val.name
        ''
        >>> fn1.return_val.doc
        'Some value'
        >>> fn1.return_val.type
        'String'

        >>> fn2 = FunctionDoc(comments[2])
        >>> fn2.return_val.doc
        'Some property of the elements.'
        >>> fn2.return_val.type
        'Array<String>'

        """
        ret = self.get('return') or self.get('returns')
        type = self.get('type')
        if '{' in ret and '}' in ret:
            if not '}  ' in ret:
                # Ensure that name is empty
                ret = ret.replace('} ', '}  ')
            return ParamDoc(ret)
        if ret and type:
            return ParamDoc('{%s}  %s' % (type, ret))
        return ParamDoc(ret)