def _initialize(self, runtime):
        """Common initializer for OsidManager and OsidProxyManager"""

        if runtime is None:
            raise NullArgument()
        if self._my_runtime is not None:
            raise IllegalState('this manager has already been initialized.')
        self._my_runtime = runtime
        config = runtime.get_configuration()

        cf_public_key_param_id = Id('parameter:cloudFrontPublicKey@aws_adapter')
        cf_private_key_param_id = Id('parameter:cloudFrontPrivateKey@aws_adapter')
        cf_keypair_id_param_id = Id('parameter:cloudFrontSigningKeypairId@aws_adapter')
        cf_private_key_file_param_id = Id('parameter:cloudFrontSigningPrivateKeyFile@aws_adapter')
        cf_distro_param_id = Id('parameter:cloudFrontDistro@aws_adapter')
        cf_distro_id_param_id = Id('parameter:cloudFrontDistroId@aws_adapter')
        s3_public_key_param_id = Id('parameter:S3PublicKey@aws_adapter')
        s3_private_key_param_id = Id('parameter:S3PrivateKey@aws_adapter')
        s3_bucket_param_id = Id('parameter:S3Bucket@aws_adapter')

        cf_public_key = config.get_value_by_parameter(cf_public_key_param_id).get_string_value()
        cf_private_key = config.get_value_by_parameter(cf_private_key_param_id).get_string_value()
        cf_keypair_id = config.get_value_by_parameter(cf_keypair_id_param_id).get_string_value()
        cf_private_key_file = config.get_value_by_parameter(
            cf_private_key_file_param_id).get_string_value()
        cf_distro = config.get_value_by_parameter(cf_distro_param_id).get_string_value()
        cf_distro_id = config.get_value_by_parameter(cf_distro_id_param_id).get_string_value()
        s3_public_key = config.get_value_by_parameter(s3_public_key_param_id).get_string_value()
        s3_private_key = config.get_value_by_parameter(s3_private_key_param_id).get_string_value()
        s3_bucket = config.get_value_by_parameter(s3_bucket_param_id).get_string_value()

        self._config_map['cloudfront_public_key'] = cf_public_key
        self._config_map['cloudfront_private_key'] = cf_private_key
        self._config_map['cloudfront_keypair_id'] = cf_keypair_id
        self._config_map['cloudfront_private_key_file'] = cf_private_key_file
        self._config_map['cloudfront_distro'] = cf_distro
        self._config_map['cloudfront_distro_id'] = cf_distro_id
        self._config_map['put_public_key'] = s3_public_key
        self._config_map['put_private_key'] = s3_private_key
        self._config_map['s3_bucket'] = s3_bucket