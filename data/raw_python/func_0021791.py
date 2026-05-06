def raise_for_api_error(headers: MutableMapping, data: MutableMapping) -> None:
    """
    Check request response for Slack API error

    Args:
        headers: Response headers
        data: Response data

    Raises:
        :class:`slack.exceptions.SlackAPIError`
    """

    if not data["ok"]:
        raise exceptions.SlackAPIError(data.get("error", "unknow_error"), headers, data)

    if "warning" in data:
        LOG.warning("Slack API WARNING: %s", data["warning"])