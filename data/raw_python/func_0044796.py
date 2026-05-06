def _generate_endpoint(self, ep_name, ep_method):
        """
        Generate configuration for a single endpoint (this is many resources)

        Terraform Names:

        - aws_api_gateway_resource: {ep_name}
        - aws_api_gateway_method: {ep_name}_{ep_method}

        :param ep_name: endpoint name (path component)
        :type ep_name: str
        :param ep_method: HTTP method for the endpoint
        :type ep_method: str
        """
        ep_method = ep_method.upper()
        self.tf_conf['resource']['aws_api_gateway_resource'][ep_name] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'parent_id':
                '${aws_api_gateway_rest_api.rest_api.root_resource_id}',
            'path_part': ep_name
        }
        self.tf_conf['output']['%s_path' % ep_name] = {
            'value': '${aws_api_gateway_resource.%s.path}' % ep_name
        }
        self.tf_conf['resource']['aws_api_gateway_method'][
            '%s_%s' % (ep_name, ep_method)] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'resource_id': '${aws_api_gateway_resource.%s.id}' % ep_name,
            'http_method': ep_method,
            'authorization': 'NONE',
            # @TODO: request_models ?
            # @TODO: request_parameters_in_json ?
        }
        self.tf_conf['resource']['aws_api_gateway_method_response'][
            '%s_%s_202' % (ep_name, ep_method)] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'resource_id': '${aws_api_gateway_resource.%s.id}' % ep_name,
            'http_method': ep_method,
            'status_code': 202,
            'response_models': {
                'application/json':
                    '${aws_api_gateway_model.successmessage.name}',
            },
            'depends_on': [
                'aws_api_gateway_method.%s_%s' % (ep_name, ep_method)
            ]
        }
        self.tf_conf['resource']['aws_api_gateway_method_response'][
            '%s_%s_500' % (ep_name, ep_method)] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'resource_id': '${aws_api_gateway_resource.%s.id}' % ep_name,
            'http_method': ep_method,
            'status_code': 500,
            'response_models': {
                'application/json':
                    '${aws_api_gateway_model.errormessage.name}',
            },
            'depends_on': [
                'aws_api_gateway_method.%s_%s' % (ep_name, ep_method)
            ]
        }

        self.tf_conf['resource']['aws_api_gateway_integration'][
            '%s_%s_integration' % (ep_name, ep_method)] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'resource_id': '${aws_api_gateway_resource.%s.id}' % ep_name,
            'http_method': ep_method,
            'type': 'AWS',
            'uri': 'arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/'
                   'functions/${aws_lambda_function.lambda_func.arn}'
                   '/invocations',
            'credentials': '${aws_iam_role.invoke_role.arn}',
            'integration_http_method': 'POST',
            'request_templates': request_model_mapping
            # @TODO:
            # request_parameters_in_json
            # integrationResponses
        }

        self.tf_conf['resource']['aws_api_gateway_integration_response'][
            '%s_%s_successResponse' % (ep_name, ep_method)] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'resource_id': '${aws_api_gateway_resource.%s.id}' % ep_name,
            'http_method': ep_method,
            'status_code': 202,
            'response_templates': response_model_mapping['success'],
            'depends_on': [
                'aws_api_gateway_method_response.%s_%s_202' % (
                    ep_name, ep_method),
                'aws_api_gateway_integration.%s_%s_integration' % (
                    ep_name, ep_method)
            ]
        }
        self.tf_conf['resource']['aws_api_gateway_integration_response'][
            '%s_%s_errorResponse' % (ep_name, ep_method)] = {
            'rest_api_id': '${aws_api_gateway_rest_api.rest_api.id}',
            'resource_id': '${aws_api_gateway_resource.%s.id}' % ep_name,
            'http_method': ep_method,
            'status_code': 500,
            'selection_pattern': '(^Failed.*)|(.*([Ee]xception|[Ee]rror).*)',
            'response_templates': response_model_mapping['error'],
            'depends_on': [
                'aws_api_gateway_method_response.%s_%s_500' % (
                    ep_name, ep_method),
                'aws_api_gateway_integration.%s_%s_integration' % (
                    ep_name, ep_method)
            ]
        }