def audit_2_3(self):
        """2.3 Ensure the S3 bucket CloudTrail logs to is not publicly accessible (Scored)"""
        raise NotImplementedError()
        import boto3
        s3 = boto3.session.Session(region_name="us-east-1").resource("s3")
        # s3 = boto3.resource("s3")
        # for trail in self.trails:
        #    for grant in s3.Bucket(trail["S3BucketName"]).Acl().grants:
        #    print(s3.Bucket(trail["S3BucketName"]).Policy().policy)
        for bucket in s3.buckets.all():
            print(bucket)
            try:
                print("    Policy:", bucket.Policy().policy)
            except Exception:
                pass
            for grant in bucket.Acl().grants:
                try:
                    print("    Grant:", grant)
                except Exception:
                    pass