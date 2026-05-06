def check_token(func):
    """检查 access token 是否有效."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        if response.status_code == 401:
            raise InvalidToken('Access token invalid or no longer valid')
        else:
            return response
    return wrapper