def create_param_doc(cls, param, prefix=None):
        """
        Generate documentation for single parameter of function
        :param param: dict contains info about parameter
        :param sub: prefix string for recursive purposes
        """
        desc = cls.exclude_html_reg.sub('', param['description']).strip()
        if not desc:
            desc = "<no description>"
        name = param['name']
        if prefix:
            name = "%s[%s]" % (prefix, name)
        doc_ = ":param %s: %s; %s" % (name, desc, param['validator'])
        if param['required']:
            doc_ += " (REQUIRED)"
        else:
            doc_ += " (OPTIONAL)"
        for param in param.get('params', []):
            doc_ += "\n" + cls.create_param_doc(param, name)
        return doc_