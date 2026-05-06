def generate_func(self, as_global=False):
        """
        Generate function for specific method and using specific api

        :param as_global: if set, will use the global function name, instead of
            the class method (usually {resource}_{class_method}) when defining
            the function
        """
        keywords = []
        params_def = []
        params_doc = ""
        original_names = {}

        params = dict(
            (param['name'], param)
            for param in self.params
        )

        # parse the url required params, as sometimes they are skipped in the
        # parameters list of the definition
        for param in self.url_params:
            if param not in params:
                param = {
                    'name': param,
                    'required': True,
                    'description': '',
                    'validator': '',
                }
                params[param['name']] = param
            else:
                params[param]['required'] = True

        # split required and non-required params for the definition
        req_params = []
        nonreq_params = []
        for param in six.itervalues(params):
            if param['required']:
                req_params.append(param)
            else:
                nonreq_params.append(param)

        for param in req_params + nonreq_params:
            params_doc += self.create_param_doc(param) + "\n"
            local_name = param['name']
            # some params collide with python keywords, that's why we do
            # this switch (and undo it inside the function we generate)
            if param['name'] == 'except':
                local_name = 'except_'
            original_names[local_name] = param['name']
            keywords.append(local_name)
            if param['required']:
                params_def.append("%s" % local_name)
            else:
                params_def.append("%s=None" % local_name)

        func_head = 'def {0}(self, {1}):'.format(
            as_global and self.get_global_method_name() or self.name,
            ', '.join(params_def)
        )
        code_body = (
            '   _vars_ = locals()\n'
            '   _url = self._fill_url("{url}", _vars_, {url_params})\n'
            '   _original_names = {original_names}\n'
            '   _kwargs = dict((_original_names[k], _vars_[k])\n'
            '                   for k in {keywords} if _vars_[k])\n'
            '   return self._foreman.do_{http_method}(_url, _kwargs)')
        code_body = code_body.format(
            http_method=self.http_method.lower(),
            url=self.url,
            url_params=self.url_params,
            keywords=keywords,
            original_names=original_names,
        )

        code = [
            func_head,
            '   """',
            self.short_desc,
            '',
            params_doc,
            '   """',
            code_body,
        ]

        code = '\n'.join(code)

        six.exec_(code)

        function = locals()[self.name]
        # to ease debugging, all the funcs have the definitions attached
        setattr(function, 'defs', self)
        return function