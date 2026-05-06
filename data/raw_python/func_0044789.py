def _generate_iam_invoke_role_policy(self):
        """
        Generate the policy for the IAM role used by API Gateway to invoke
        the lambda function.

        Terraform name: aws_iam_role.invoke_role
        """
        invoke_pol = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Resource": ["*"],
                    "Action": ["lambda:InvokeFunction"]
                }
            ]
        }
        self.tf_conf['resource']['aws_iam_role_policy']['invoke_policy'] = {
            'name': self.resource_name + '-invoke',
            'role': '${aws_iam_role.invoke_role.id}',
            'policy': json.dumps(invoke_pol)
        }