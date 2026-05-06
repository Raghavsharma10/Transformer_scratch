def is_valid_ip(address):
    """Verifica se address é um endereço ip válido.

    O valor é considerado válido se tiver no formato XXX.XXX.XXX.XXX, onde X é um valor entre 0 e 9.

    :param address: Endereço IP.

    :return: True se o parâmetro é um IP válido, ou False, caso contrário.
    """
    if address is None:
        return False
    pattern = r'\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    return re.match(pattern, address)