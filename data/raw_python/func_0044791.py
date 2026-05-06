def _generate_iam_invoke_role(self):
        """
        Generate the IAM Role for API Gateway to use to invoke the function.

        Terraform name: aws_iam_role.invoke_role
        :return:
        """

        invoke_assume = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "sts:AssumeRole",
                    "Principal": {
                        "Service": "apigateway.amazonaws.com"
                    },
                    "Effect": "Allow",
                    "Sid": ""
                }
            ]
        }
        self.tf_conf['resource']['aws_iam_role']['invoke_role'] = {
            'name': self.resource_name + '-invoke',
            'assume_role_policy': json.dumps(invoke_assume),
        }
        self.tf_conf['output']['iam_invoke_role_arn'] = {
            'value': '${aws_iam_role.invoke_role.arn}'
        }
        self.tf_conf['output']['iam_invoke_role_unique_id'] = {
            'value': '${aws_iam_role.invoke_role.unique_id}'
        }