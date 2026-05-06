def translate_ref_to_url(self, ref, in_comment=None):
        """
        Translates an @see or @link reference to a URL.  If the ref is of the 
        form #methodName, it looks for a method of that name on the class
        `in_comment` or parent class of method `in_comment`.  In this case, it
        returns a local hash URL, since the method is guaranteed to be on the
        same page:

        >>> doc = CodeBaseDoc(['examples'])
        >>> doc.translate_ref_to_url('#public_method', doc.all_methods['private_method'])
        '#public_method'
        >>> doc.translate_ref_to_url('#public_method', doc.all_classes['MySubClass'])
        '#public_method'

        If it doesn't find it there, it looks for a global function:

        >>> doc.translate_ref_to_url('#make_class')
        'module_closure.html#make_class'

        A reference of the form ClassName#method_name looks up a specific method:

        >>> doc.translate_ref_to_url('MyClass#first_method')
        'class.html#first_method'

        Finally, a reference of the form ClassName looks up a specific class:

        >>> doc.translate_ref_to_url('MyClass')
        'class.html#MyClass'

        """
        if ref.startswith('#'):
            method_name = ref[1:]
            if isinstance(in_comment, FunctionDoc) and in_comment.member:
                search_in = self.all_classes[in_comment.member]
            elif isinstance(in_comment, ClassDoc):
                search_in = in_comment
            else:
                search_in = None

            try:
                return search_in.get_method(method_name).url
            except AttributeError:
                pass

            def lookup_ref(file_doc):
                for fn in file_doc.functions:
                    if fn.name == method_name:
                        return fn.url
                return None
        elif '#' in ref:
            class_name, method_name = ref.split('#')
            def lookup_ref(file_doc):
                for cls in file_doc.classes:
                    if cls.name == class_name:
                        try:
                            return cls.get_method(method_name).url
                        except AttributeError:
                            pass
                return None
        else:
            class_name = ref
            def lookup_ref(file_doc):
                for cls in file_doc.classes:
                    if cls.name == class_name:
                        return cls.url
                return None

        for file_doc in list(self.values()):
            url = lookup_ref(file_doc)
            if url:
                return file_doc.url + url
        return ''