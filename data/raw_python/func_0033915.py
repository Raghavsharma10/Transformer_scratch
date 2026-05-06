def build_artefact(self, src_project=None):
        """Run deploy the lambdas defined in our project.
        Steps:
        * Build Artefact
        * Read file or deploy to S3. It's defined in config["deploy"]["deploy_method"]

        :param src_project: str. Name of the folder or path of the project where our code lives
        :return: bool
        """
        path_to_zip_file = self.build.run(src_project or self.config.get_projectdir())
        self.set_artefact_path(path_to_zip_file)

        deploy_method = self.config["deploy"]["deploy_method"]

        if deploy_method == "S3":
            deploy_bucket = self.config["deploy"]["deploy_bucket"]
            bucket = self.awss3.Bucket(deploy_bucket)
            try:
                self.awss3.meta.client.head_bucket(Bucket=deploy_bucket)
            except ClientError as e:
                if e.response['Error']['Code'] == "404" or e.response['Error']['Code'] == "NoSuchBucket":
                    region = self.config.get("aws_credentials", {}).get("region", None)

                    logger.info(
                        "Bucket not exist. Creating new one with name {} in region {}".format(deploy_bucket, region))
                    bucket_conf = {}
                    if region:
                        bucket_conf = {"CreateBucketConfiguration": {'LocationConstraint': region}}
                    bucket.wait_until_not_exists()
                    bucket.create(**bucket_conf)
                    bucket.wait_until_exists()

                else:
                    # TODO: handle other errors there
                    pass
            s3_keyfile = self.config["deploy"]["deploy_file"].split(os.path.sep)[-1]
            bucket.put_object(
                Key=s3_keyfile,
                Body=self.build.read(self.config["deploy"]["deploy_file"])

            )
            code = {'S3Bucket': deploy_bucket, 'S3Key': s3_keyfile, }

        elif deploy_method == "FILE":
            code = {'ZipFile': self.build.read(self.config["deploy"]["deploy_file"])}
        else:
            raise Exception("No deploy_method in config")

        return code