def decode_iter_request(data: dict) -> Optional[Union[str, int]]:
    """
    Decode incoming response from an iteration request

    Args:
        data: Response data

    Returns:
        Next itervalue
    """
    if "response_metadata" in data:
        return data["response_metadata"].get("next_cursor")
    elif "paging" in data:
        current_page = int(data["paging"].get("page", 1))
        max_page = int(data["paging"].get("pages", 1))

        if current_page < max_page:
            return current_page + 1
    elif "has_more" in data and data["has_more"] and "latest" in data:
        return data["messages"][-1]["ts"]

    return None