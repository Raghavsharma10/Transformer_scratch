def is_valid_int_param(param):
    """Verifica se o parâmetro é um valor inteiro válido.

    :param param: Valor para ser validado.

    :return: True se o parâmetro tem um valor inteiro válido, ou False, caso contrário.
    """
    if param is None:
        return False
    try:
        param = int(param)
        if param < 0:
            return False
    except (TypeError, ValueError):
        return False
    return True