def configure(username, password):
    # type: (str, str) -> None
    """
    Generate .pypirc config with the given credentials.

    Example:

        $ peltak pypi configure my_pypi_user my_pypi_pass

    """
    from peltak.extra.pypi import logic
    logic.gen_pypirc(username, password)