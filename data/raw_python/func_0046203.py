def methods(self):
        """
        Returns a generator of all member functions in the file, in textual
        order.  

        >>> file = FileDoc('class.js', read_file('examples/class.js'))
        >>> file.methods.next().name
        'first_method'

        """
        def is_method(comment):
            return isinstance(comment, FunctionDoc) and comment.member
        return self._filtered_iter(is_method)