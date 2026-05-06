def _assign_method(self, resource_class, method_type):
        """
        Using reflection, assigns a new method to this class.

        Args:
            resource_class: A resource class
            method_type: The HTTP method type
        """

        """
        If we assigned the same method to each method, it's the same
        method in memory, so we need one for each acceptable HTTP method.
        """
        method_name = resource_class.get_method_name(
            resource_class, method_type)
        valid_status_codes = getattr(
            resource_class.Meta,
            'valid_status_codes',
            DEFAULT_VALID_STATUS_CODES
        )

        # I know what you're going to say, and I'd love help making this nicer
        # reflection assigns the same memory addr to each method otherwise.
        def get(self, method_type=method_type, method_name=method_name,
                valid_status_codes=valid_status_codes,
                resource=resource_class, data=None, uid=None, **kwargs):
            return self.call_api(
                method_type, method_name,
                valid_status_codes, resource,
                data, uid=uid, **kwargs)

        def put(self, method_type=method_type, method_name=method_name,
                valid_status_codes=valid_status_codes,
                resource=resource_class, data=None, uid=None, **kwargs):
            return self.call_api(
                method_type, method_name,
                valid_status_codes, resource,
                data, uid=uid, **kwargs)

        def post(self, method_type=method_type, method_name=method_name,
                 valid_status_codes=valid_status_codes,
                 resource=resource_class, data=None, uid=None, **kwargs):
            return self.call_api(
                method_type, method_name,
                valid_status_codes, resource,
                data, uid=uid, **kwargs)

        def patch(self, method_type=method_type, method_name=method_name,
                  valid_status_codes=valid_status_codes,
                  resource=resource_class, data=None, uid=None, **kwargs):
            return self.call_api(
                method_type, method_name,
                valid_status_codes, resource,
                data, uid=uid, **kwargs)

        def delete(self, method_type=method_type, method_name=method_name,
                   valid_status_codes=valid_status_codes,
                   resource=resource_class, data=None, uid=None, **kwargs):
            return self.call_api(
                method_type, method_name,
                valid_status_codes, resource,
                data, uid=uid, **kwargs)

        method_map = {
            'GET': get,
            'PUT': put,
            'POST': post,
            'PATCH': patch,
            'DELETE': delete
        }

        setattr(
            self, method_name,
            types.MethodType(method_map[method_type], self)
        )