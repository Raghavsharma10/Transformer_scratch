def get_crypt_key(key_path):
    """
    Get the user's PredixPy manifest key.  Generate and store one if not
    yet generated.
    """
    key_path = os.path.expanduser(key_path)
    if os.path.exists(key_path):
        with open(key_path, 'r') as data:
            key = data.read()
    else:
        key = Fernet.generate_key()
        with open(key_path, 'w') as output:
            output.write(key)

    return key