def validate_request_signature(
    body: str, headers: MutableMapping, signing_secret: str
) -> None:
    """
    Validate incoming request signature using the application signing secret.

    Contrary to the ``team_id`` and ``verification_token`` verification this method is not called by ``slack-sansio`` when creating object from incoming HTTP request. Because the body of the request needs to be provided as text and not decoded as json beforehand.

    Args:
        body: Raw request body
        headers: Request headers
        signing_secret: Application signing_secret

    Raise:
        :class:`slack.exceptions.InvalidSlackSignature`: when provided and calculated signature do not match
        :class:`slack.exceptions.InvalidTimestamp`: when incoming request timestamp is more than 5 minutes old
    """

    request_timestamp = int(headers["X-Slack-Request-Timestamp"])

    if (int(time.time()) - request_timestamp) > (60 * 5):
        raise exceptions.InvalidTimestamp(timestamp=request_timestamp)

    slack_signature = headers["X-Slack-Signature"]
    calculated_signature = (
        "v0="
        + hmac.new(
            signing_secret.encode("utf-8"),
            f"""v0:{headers["X-Slack-Request-Timestamp"]}:{body}""".encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
    )

    if not hmac.compare_digest(slack_signature, calculated_signature):
        raise exceptions.InvalidSlackSignature(slack_signature, calculated_signature)