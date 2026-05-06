def to_dict(self):
        """
        Convert this ClassDoc to a dict, such as if you want to use it in a
        template or string interpolation.  Aside from the basic `CommentDoc`
        fields, this also contains:

            - **name**: The class name
            - **method**: A list of methods, in their dictionary form.
        """
        vars = super(ClassDoc, self).to_dict()
        vars.update({
            'name': self.name,
            'method': [method.to_dict() for method in self.methods]
        })
        return vars