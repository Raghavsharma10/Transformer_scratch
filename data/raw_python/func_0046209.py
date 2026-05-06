def exceptions(self):
        """
        Returns a list of ParamDoc objects (with empty names) of the
        exception tags for the function.

        >>> comments = parse_comments_for_file('examples/module_closure.js')
        >>> fn1 = FunctionDoc(comments[1])
        >>> fn1.exceptions[0].doc
        'Another exception'
        >>> fn1.exceptions[1].doc
        'A fake exception'
        >>> fn1.exceptions[1].type
        'String'

        """
        def make_param(text):
            if '{' in text and '}' in text:
                # Make sure param name is blank:
                word_split = list(split_delimited('{}', ' ', text))
                if word_split[1] != '':
                    text = ' '.join([word_split[0], ''] + word_split[1:])
            else:
                # Handle old JSDoc format
                word_split = text.split()
                text = '{%s}  %s' % (word_split[0], ' '.join(word_split[1:]))
            return ParamDoc(text)
        return [make_param(text) for text in 
                self.get_as_list('throws') + self.get_as_list('exception')]