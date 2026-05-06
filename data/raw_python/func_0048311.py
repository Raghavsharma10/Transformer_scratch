def catch_boto_400(message, **info):
    """Turn a BotoServerError 400 into a BadAmazon"""
    try:
        yield
    except ClientError as error:
        if str(error.response["ResponseMetadata"]["HTTPStatusCode"]).startswith("4"):
            error_message = error.response["Error"]["Message"]
            raise BadAmazon(message, error_message=error_message, error_code=error.response["ResponseMetadata"]["HTTPStatusCode"], **info)
        else:
            raise