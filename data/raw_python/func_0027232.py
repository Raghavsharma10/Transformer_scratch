def list(self, request, *args, **kwargs):
        """
        To list all services without regard to its type, run **GET** against */api/services/* as an authenticated user.

        To list services of specific type issue **GET** to specific endpoint from a list above as a customer owner.
        Individual endpoint used for every service type.

        To create a service, issue a **POST** to specific endpoint from a list above as a customer owner.
        Individual endpoint used for every service type.

        You can create service based on shared service settings. Example:

        .. code-block:: http

            POST /api/digitalocean/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "name": "Common DigitalOcean",
                "customer": "http://example.com/api/customers/1040561ca9e046d2b74268600c7e1105/",
                "settings": "http://example.com/api/service-settings/93ba615d6111466ebe3f792669059cb4/"
            }

        Or provide your own credentials. Example:

        .. code-block:: http

            POST /api/oracle/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "name": "My Oracle",
                "customer": "http://example.com/api/customers/1040561ca9e046d2b74268600c7e1105/",
                "backend_url": "https://oracle.example.com:7802/em",
                "username": "admin",
                "password": "secret"
            }
        """
        return super(BaseServiceViewSet, self).list(request, *args, **kwargs)