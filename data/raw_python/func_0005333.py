def add_model_name_to_payload(cls, payload):
        """
        Checks whether the model name in question is in the payload. If not, the entire payload
        is set as a value of a key by the name of the model.  This method is useful when some
        server-side Rails API calls expect the parameters to include the parameterized model name.
        For example, server-side endpoints that handle the updating of a biosample record or the
        creation of a new biosmample record will expect the payload to be of the form::

            { "biosample": {
                "name": "new biosample",
                "donor": 3,
                ...
               }
            }

        Args:
            payload: `dict`. The data to send in an HTTP request.

        Returns:
            `dict`.
        """
        if not cls.MODEL_NAME in payload:
            payload = {cls.MODEL_NAME: payload}
        return payload