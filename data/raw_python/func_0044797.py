def _get_config(self, func_src):
        """
        Return the full terraform configuration as a JSON string

        :param func_src: lambda function source
        :type func_src: str
        :return: terraform configuration
        :rtype: str
        """
        self._set_account_info()
        self._generate_iam_role()
        self._generate_iam_role_policy()
        self._generate_iam_invoke_role()
        self._generate_iam_invoke_role_policy()
        self._generate_lambda()
        self._generate_response_models()
        self._generate_api_gateway()
        self._generate_api_gateway_deployment()
        self._generate_saved_config()
        return pretty_json(self.tf_conf)