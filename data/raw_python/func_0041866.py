def get_param_doc(doc, param):
        """Get the documentation and datatype for a parameter

        This function returns the documentation and the argument for a
        napoleon like structured docstring `doc`

        Parameters
        ----------
        doc: str
            The base docstring to use
        param: str
            The argument to use

        Returns
        -------
        str
            The documentation of the given `param`
        str
            The datatype of the given `param`"""
        arg_doc = docstrings.keep_params_s(doc, [param]) or \
            docstrings.keep_types_s(doc, [param])
        dtype = None
        if arg_doc:
            lines = arg_doc.splitlines()
            arg_doc = dedents('\n' + '\n'.join(lines[1:]))
            param_desc = lines[0].split(':', 1)
            if len(param_desc) > 1:
                dtype = param_desc[1].strip()
        return arg_doc, dtype