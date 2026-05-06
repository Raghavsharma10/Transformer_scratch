def s3(self, url, account_acessor=None, access=None, secret=None):
        """Setup an S3 pyfs, with account credentials, fixing an ssl matching problem"""
        from ambry.util.ambrys3 import AmbryS3FS
        from ambry.util import parse_url_to_dict
        import ssl

        pd = parse_url_to_dict(url)

        if account_acessor:
            account = account_acessor(pd['hostname'])

            assert account['account_id'] == pd['hostname']
            aws_access_key = account['access_key'],
            aws_secret_key = account['secret']
        else:
            aws_access_key = access
            aws_secret_key = secret

        assert access, url
        assert secret, url

        s3 = AmbryS3FS(
            bucket=pd['netloc'],
            prefix=pd['path'].strip('/')+'/',
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,

        )

        return s3