def to_dict(self):
        """
        Convert this FunctionDoc to a dictionary.  In addition to `CommentDoc`
        keys, this adds:

            - **name**: The function name
            - **params**: A list of parameter dictionaries
            - **options**: A list of option dictionaries
            - **exceptions**: A list of exception dictionaries
            - **return_val**: A dictionary describing the return type, as per `ParamDoc`
            - **is_private**: True if private
            - **is_constructor**: True if a constructor
            - **member**: The raw text of the member property.
        """
        vars = super(FunctionDoc, self).to_dict()
        vars.update({
            'name': self.name,
            'params': [param.to_dict() for param in self.params],
            'options': [option.to_dict() for option in self.options],
            'exceptions': [exc.to_dict() for exc in self.exceptions],
            'return_val': self.return_val.to_dict(),
            'is_private': self.is_private,
            'is_constructor': self.is_constructor,
            'member': self.member
        })
        return vars