def params(self):
        """
        Returns a ParamDoc for each parameter of the function, picking up
        the order from the actual parameter list.

        >>> comments = parse_comments_for_file('examples/module_closure.js')
        >>> fn2 = FunctionDoc(comments[2])
        >>> fn2.params[0].name
        'elem'
        >>> fn2.params[1].type
        'Function(DOM)'
        >>> fn2.params[2].doc
        'The Options array.'

        """
        tag_texts = self.get_as_list('param') + self.get_as_list('argument')
        if self.get('guessed_params') is None:
            return [ParamDoc(text) for text in tag_texts]
        else:
            param_dict = {}
            for text in tag_texts:
                param = ParamDoc(text)
                param_dict[param.name] = param
            return [param_dict.get(name) or ParamDoc('{} ' + name)
                    for name in self.get('guessed_params')]