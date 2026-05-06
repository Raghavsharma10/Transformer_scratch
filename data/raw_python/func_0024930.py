def _init_channel(self):
        """
        build the grpc channel used for both publisher and subscriber
        :return: None
        """
        host = self._get_host()
        port = self._get_grpc_port()

        if 'TLS_PEM_FILE' in os.environ:
            with open(os.environ['TLS_PEM_FILE'], mode='rb') as f:  # b is important -> binary
                file_content = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates=file_content)
        else:
            credentials = grpc.ssl_channel_credentials()

        self._channel = grpc.secure_channel(host + ":" + port, credentials=credentials)
        self._init_health_checker()