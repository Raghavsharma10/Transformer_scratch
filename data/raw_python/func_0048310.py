def catch_no_credentials(message, **info):
    """Turn a NoCredentialsError into a BadAmazon"""
    try:
        yield
    except NoCredentialsError as error:
        if hasattr(error, "response"):
            info['error_code'] = error.response["ResponseMetadata"]["HTTPStatusCode"]
            info['error_message'] = error.response["Error"]["Message"]
        else:
            info['error_message'] = error.fmt

        raise BadAmazon(message, **info)