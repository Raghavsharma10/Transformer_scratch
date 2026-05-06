def get_method_name(resource, method_type):
        """
        Generate a method name for this resource based on the method type.
        """
        return '{}_{}'.format(method_type.lower(), resource.Meta.name.lower())