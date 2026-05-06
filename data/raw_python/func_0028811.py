def launch_exception(message):
    """
        Launch a Python exception from an error that took place in the browser.

        messsage format:
        - name: str
        - description: str
    """
    error_name = message['name']
    error_descr = message['description']
    mapping = {
        'ReferenceError': NameError,
    }
    if message['name'] in mapping:
        raise mapping[error_name](error_descr)
    else:
        raise Exception('{}: {}'.format(error_name, error_descr))