def hms(segundos):  # TODO: mover para util.py
    """
    Retorna o número de horas, minutos e segundos a partir do total de
    segundos informado.

    .. sourcecode:: python

        >>> hms(1)
        (0, 0, 1)

        >>> hms(60)
        (0, 1, 0)

        >>> hms(3600)
        (1, 0, 0)

        >>> hms(3601)
        (1, 0, 1)

        >>> hms(3661)
        (1, 1, 1)

    :param int segundos: O número total de segundos.

    :returns: Uma tupla contendo trẽs elementos representando, respectivamente,
        o número de horas, minutos e segundos calculados a partir do total de
        segundos.

    :rtype: tuple
    """
    h = (segundos / 3600)
    m = (segundos - (3600 * h)) / 60
    s = (segundos - (3600 * h) - (m * 60));
    return (h, m, s)