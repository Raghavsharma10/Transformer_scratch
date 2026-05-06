def parse_content_type(headers: MutableMapping) -> Tuple[Optional[str], str]:
    """
    Find content-type and encoding of the response

    Args:
        headers: Response headers

    Returns:
        :py:class:`tuple` (content-type, encoding)
    """
    content_type = headers.get("content-type")
    if not content_type:
        return None, "utf-8"
    else:
        type_, parameters = cgi.parse_header(content_type)
        encoding = parameters.get("charset", "utf-8")
        return type_, encoding