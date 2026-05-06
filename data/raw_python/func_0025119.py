def hms_humanizado(segundos): # TODO: mover para util.py
    """
    Retorna um texto legível que descreve o total de horas, minutos e segundos
    calculados a partir do total de segundos informados.

    .. sourcecode:: python

        >>> hms_humanizado(0)
        'zero segundos'

        >>> hms_humanizado(1)
        '1 segundo'

        >>> hms_humanizado(2)
        '2 segundos'

        >>> hms_humanizado(3600)
        '1 hora'

        >>> hms_humanizado(3602)
        '1 hora e 2 segundos'

        >>> hms_humanizado(3721)
        '1 hora, 2 minutos e 1 segundo'

    :rtype: str
    """
    p = lambda n, s, p: p if n > 1 else s
    h, m, s = hms(segundos)

    tokens = [
            '' if h == 0 else '{:d} {}'.format(h, p(h, 'hora', 'horas')),
            '' if m == 0 else '{:d} {}'.format(m, p(m, 'minuto', 'minutos')),
            '' if s == 0 else '{:d} {}'.format(s, p(s, 'segundo', 'segundos'))]

    tokens = [token for token in tokens if token]

    if len(tokens) == 1:
        return tokens[0]

    if len(tokens) > 1:
        return '{} e {}'.format(', '.join(tokens[:-1]), tokens[-1])

    return 'zero segundos'