def to_dash_case(string: str) -> str:
    """
    Convert a string to dash-delimited words.

    ::

        >>> import uqbar.strings
        >>> string = 'Tô Đặc Biệt Xe Lửa'
        >>> print(uqbar.strings.to_dash_case(string))
        to-dac-biet-xe-lua

    ::

        >>> string = 'alpha.beta.gamma'
        >>> print(uqbar.strings.to_dash_case(string))
        alpha-beta-gamma

    """
    string = unidecode.unidecode(string)
    words = (_.lower() for _ in delimit_words(string))
    string = "-".join(words)
    return string