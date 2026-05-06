def get_proxy(self, input_):
        """Gets a proxy.

        arg:    input (osid.proxy.ProxyCondition): a proxy condition
        return: (osid.proxy.Proxy) - a proxy
        raise:  NullArgument - ``input`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``input`` is not of this service
        *compliance: mandatory -- This method is must be implemented.*

        """
        if input_._http_request is not None:
            authentication = Authentication()
            authentication.set_django_user(input_._http_request.user)
        else:
            authentication = None
        effective_agent_id = input_._effective_agent_id
        # Also need to deal with effective dates and Local
        return rules.Proxy(authentication=authentication,
                           effective_agent_id=effective_agent_id)