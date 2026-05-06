def negotiate(cls, headers):
        """ Process headers dict to return the format class
            (not the instance)
        """
        # set lower keys
        headers = {k.lower(): v for k, v in headers.items()}

        accept = headers.get('accept', "*/*")

        parsed_accept = accept.split(";")
        parsed_accept = [i.strip() for i in parsed_accept]

        # Protobuffer (only one version)
        if all([i in parsed_accept for i in cls.PROTOBUF['default']]):
            return ProtobufFormat
        elif all([i in parsed_accept for i in cls.PROTOBUF['text']]):
            return ProtobufTextFormat
        # Text 0.0.4
        elif all([i in parsed_accept for i in cls.TEXT['0.0.4']]):
            return TextFormat
        # Text (Default)
        elif all([i in parsed_accept for i in cls.TEXT['default']]):
            return TextFormat
        # Default
        else:
            return cls.FALLBACK