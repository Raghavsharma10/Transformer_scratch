def functions(self):
        """
        Returns a generator of all standalone functions in the file, in textual
        order.

        >>> file = FileDoc('module.js', read_file('examples/module.js'))
        >>> list(file.functions)[0].name
        'the_first_function'
        >>> list(file.functions)[3].name
        'not_auto_discovered'

        """
        def is_function(comment):
            return isinstance(comment, FunctionDoc) and not comment.member
        return self._filtered_iter(is_function)