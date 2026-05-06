def add(cls, code_name, name='', description=''):
        """
        create a custom permission
        """
        if code_name not in cls.registry:
            cls.registry[code_name] = (code_name, name or code_name, description)
        return code_name