def get_api_envs():
    """Get required API keys from environment variables."""
    client_id = os.environ.get('CLIENT_ID')
    user_id = os.environ.get('USER_ID')
    if not client_id or not user_id:
        raise ValueError('API keys are not found in the environment')
    return client_id, user_id