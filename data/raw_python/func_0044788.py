def _generate_iam_role_policy(self):
        """
        Generate the policy for the IAM Role.

        Terraform name: aws_iam_role.lambda_role
        """
        endpoints = self.config.get('endpoints')
        queue_arns = []
        for ep in endpoints:
            for qname in endpoints[ep]['queues']:
                qarn = 'arn:aws:sqs:%s:%s:%s' % (self.aws_region,
                                                 self.aws_account_id, qname)
                if qarn not in queue_arns:
                    queue_arns.append(qarn)
        pol = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "logs:CreateLogGroup",
                    "Resource": "arn:aws:logs:%s:%s:*" % (
                        self.aws_region, self.aws_account_id
                    )
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": [
                        "arn:aws:logs:%s:%s:log-group:%s:*" % (
                            self.aws_region, self.aws_account_id,
                            '/aws/lambda/%s' % self.resource_name
                        )
                    ]
                },
                {
                    'Effect': 'Allow',
                    'Action': [
                        'sqs:ListQueues'
                    ],
                    'Resource': '*'
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "sqs:GetQueueUrl",
                        "sqs:SendMessage"
                    ],
                    "Resource": sorted(queue_arns)
                }
            ]
        }
        self.tf_conf['resource']['aws_iam_role_policy']['role_policy'] = {
            'name': self.resource_name,
            'role': '${aws_iam_role.lambda_role.id}',
            'policy': json.dumps(pol)
        }