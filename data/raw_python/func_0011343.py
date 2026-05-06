def stringify(req, resp):
    """
    dumps all valid jsons
    This is the latest after hook
    """
    if isinstance(resp.body, dict):
        try:
            resp.body = json.dumps(resp.body)
        except(nameError):
            resp.status = falcon.HTTP_500