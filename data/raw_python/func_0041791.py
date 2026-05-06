def python(string: str):
        """
            :param string: String can be type, resource or python case
        """
        return underscore(singularize(string) if Naming._pluralize(string) else string)