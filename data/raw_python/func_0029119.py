def build_policy_bucket(self, bucket, name, statements):
        """
        Generate bucket policy for S3 bucket
        :param bucket: The bucket to attach policy to
        :param name: The name of the bucket (to generate policy name from it)
        :param statements: The "rules" the policy should have
        :return: Ref to new policy
        """

        policy = self.__template.add_resource(
            BucketPolicy(
                self.name_strip(name, True, False),
                Bucket=troposphere.Ref(bucket),
                DependsOn=[
                    troposphere.Name(bucket)
                ],
                PolicyDocument=Policy(
                    Version=self.VERSION_IAM,
                    Statement=statements
                )
            )
        )

        return policy