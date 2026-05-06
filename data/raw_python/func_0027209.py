def create(self, request, *args, **kwargs):
        """
        A new customer can only be created:

         - by users with staff privilege (is_staff=True);
         - by organization owners if OWNER_CAN_MANAGE_CUSTOMER is set to True;

        Example of a valid request:

        .. code-block:: http

            POST /api/customers/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "name": "Customer A",
                "native_name": "Customer A",
                "abbreviation": "CA",
                "contact_details": "Luhamaa 28, 10128 Tallinn",
            }
        """
        return super(CustomerViewSet, self).create(request, *args, **kwargs)