def build_bucket(self, name, lifecycle_configuration=False,
                     use_plain_name=False):
        """
        Generate S3 bucket statement
        :param name: Name of the bucket
        :param lifecycle_configuration: Additional lifecycle configuration (default=False)
        :param use_plain_name: Just use the given name and do not add prefix
        :return: Ref to new bucket
        """
        if use_plain_name:
            name_aws = name_bucket = name
            name_aws = name_aws.title()
            name_aws = name_aws.replace('-', '')
        else:
            name_aws = self.name_strip(name, False, False)
            name_bucket = self.name_build(name)

        if lifecycle_configuration:
            return self.__template.add_resource(
                Bucket(
                    name_aws,
                    BucketName=name_bucket,
                    LifecycleConfiguration=lifecycle_configuration
                )
            )
        else:
            return self.__template.add_resource(
                Bucket(
                    name_aws,
                    BucketName=name_bucket,
                )
            )