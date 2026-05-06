def _generate_api_gateway(self):
        """
        Generate the full configuration for the API Gateway, and add to
        self.tf_conf
        """
        self.tf_conf['resource']['aws_api_gateway_rest_api']['rest_api'] = {
            'name': self.resource_name,
            'description': self.description
        }
        self.tf_conf['output']['rest_api_id'] = {
            'value': '${aws_api_gateway_rest_api.rest_api.id}'
        }
        # finally, the deployment
        """
        @NOTE Currently, Terraform can't enable metrics collection,
        request logging or rate limiting on API Gateway services.

        @TODO update this when
        <https://github.com/hashicorp/terraform/issues/6612> is fixed.

        @see https://github.com/jantman/webhook2lambda2sqs/issues/7
        @see https://github.com/jantman/webhook2lambda2sqs/issues/16
        """
        self.tf_conf['output']['base_url'] = {
            'value': 'https://${aws_api_gateway_rest_api.rest_api.id}.'
                     'execute-api.%s.amazonaws.com/%s/' % (
                         self.aws_region, self.config.stage_name)
        }
        # generate the endpoint configs
        endpoints = self.config.get('endpoints')
        for ep in sorted(endpoints.keys()):
            self._generate_endpoint(ep, endpoints[ep]['method'])