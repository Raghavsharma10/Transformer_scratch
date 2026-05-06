def get_signed_url(url, config_map):
    """ Convenience function for getting cloudfront signed URL given a saved URL

    cjshaw, Jan 7, 2015

    Follows:
        http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/
            private-content-creating-signed-url-canned-policy.html#private-
            content-creating-signed-url-canned-policy-procedure
        http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/
            PrivateContent.html
        http://boto.readthedocs.org/en/latest/ref/cloudfront.html

    May 25, 2017: Switch to boto3

    """
    # From https://stackoverflow.com/a/34322915
    def rsa_signer(message):
        private_key = open(config_map['cloudfront_private_key_file'], 'r').read()
        return rsa.sign(
            message,
            rsa.PrivateKey.load_pkcs1(private_key.encode('utf8')),
            'SHA-1')  # CloudFront requires SHA-1 hash

    if any(config_map[key] == '' for key in ['s3_bucket', 'cloudfront_distro',
                                             'cloudfront_private_key_file', 'cloudfront_keypair_id']):
        # This is a test configuration
        return 'You are missing S3 and CF configs: https:///?Expires=X&Signature=X&Key-Pair-Id='

    expires = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    s3_bucket = config_map['s3_bucket']

    url = url.replace(s3_bucket + '.s3.amazonaws.com', config_map['cloudfront_distro'])

    if not AWS_CLIENT.is_aws_cf_client_set():
        cf_signer = CloudFrontSigner(
            config_map['cloudfront_keypair_id'],
            rsa_signer
        )
        AWS_CLIENT.set_aws_cf_client(cf_signer)
    else:
        cf_signer = AWS_CLIENT.cf
    signed_url = cf_signer.generate_presigned_url(
        url,
        date_less_than=expires)
    return signed_url