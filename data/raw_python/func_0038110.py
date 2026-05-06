def url_split(s3_url):
    """Split S3 URL and return a tuple of (bucket, key).

    S3 URL is expected to be of "s3://<bucket>/<key>" format.
    """

    assert isinstance(s3_url, str)

    match = re_s3_url.match(s3_url)
    if not match:
        raise UrlParseError('Error parsing S3 URL: "%s"' % s3_url)

    parts = match.groupdict()
    return (parts['bucket'], parts['key'])