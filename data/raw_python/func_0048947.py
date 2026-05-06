def from_status(cls, http_status, code_index=0, message=None, developer_message=None, meta=None):
        # type: (HTTPStatus, int, AnyStr, AnyStr, dict) -> Error
        """
        Automatically build an HTTP response from the HTTP Status code.
        
        :param http_status: 
        :param code_index: 
        :param message: 
        :param developer_message: 
        :param meta: 

        """
        return cls(http_status.value,
                   (http_status.value * 100) + code_index,
                   message or http_status.description,
                   developer_message or http_status.description,
                   meta)