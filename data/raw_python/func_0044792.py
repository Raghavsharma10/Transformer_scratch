def _generate_lambda(self):
        """
        Generate the lambda function and its IAM role, and add to self.tf_conf
        """
        self.tf_conf['resource']['aws_lambda_function']['lambda_func'] = {
            'filename': 'webhook2lambda2sqs_func.zip',
            'function_name': self.resource_name,
            'role': '${aws_iam_role.lambda_role.arn}',
            'handler': 'webhook2lambda2sqs_func.webhook2lambda2sqs_handler',
            'source_code_hash': '${base64sha256(file('
                                '"webhook2lambda2sqs_func.zip"))}',
            'description': self.description,
            'runtime': 'python2.7',
            'timeout': 120
        }
        self.tf_conf['output']['lambda_func_arn'] = {
            'value': '${aws_lambda_function.lambda_func.arn}'
        }