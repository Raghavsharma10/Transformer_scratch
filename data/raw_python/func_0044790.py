def _generate_iam_role(self):
        """
        Generate the IAM Role needed by the Lambda function.

        Terraform name: aws_iam_role.lambda_role
        """
        pol = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "sts:AssumeRole",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Effect": "Allow",
                    "Sid": ""
                }
            ]
        }

        self.tf_conf['resource']['aws_iam_role'] = {}
        self.tf_conf['resource']['aws_iam_role']['lambda_role'] = {
            'name': self.resource_name,
            'assume_role_policy': json.dumps(pol),
        }
        self.tf_conf['output']['iam_role_arn'] = {
            'value': '${aws_iam_role.lambda_role.arn}'
        }
        self.tf_conf['output']['iam_role_unique_id'] = {
            'value': '${aws_iam_role.lambda_role.unique_id}'
        }