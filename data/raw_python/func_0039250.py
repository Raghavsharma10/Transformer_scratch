def passgen(length=12, punctuation=False, digits=True, letters=True,
            case="both", **kwargs):
    """Generate random password.

    Args:
        length (int): The length of the password.  Must be greater than
            zero. Defaults to 12.
        punctuation (bool): Whether to use punctuation or not.  Defaults
            to False.
        limit_punctuation (str): Limits the allowed puncturation to defined 
            characters.
        digits (bool): Whether to use digits or not.  Defaults to True.
            One of *digits* and *letters* must be True.
        letters (bool): Whether to use letters or not.  Defaults to
            True. One of *digits* and *letters* must be True.
        case (str): Letter case to use.  Accepts 'upper' for upper case,
            'lower' for lower case, and 'both' for both.  Defaults to
            'both'.
        
    Returns:
        str. The generated password.

    Raises:
        ValueError

    Below are some basic examples.

    >>> passgen()
    z7GlutdEEbnk

    >>> passgen(case='upper')
    Q81J9DOAMBRN

    >>> passgen(length=6)
    EzJMRX
    """

    p_min = punctuation
    p_max = 0 if punctuation is False else length
    d_min = digits
    d_max = 0 if digits is False else length
    a_min = letters
    a_max = 0 if letters is False else length

    if d_min + p_min + a_min > length:
        raise ValueError("Minimum punctuation and digits number cannot be greater than length")    
    if not digits and not letters:
        raise ValueError("digits and letters cannot be False at the same time")
    if length < 1:
        raise ValueError("length must be greater than zero")

    if letters:
        if case == "both":
            alpha = string.ascii_uppercase + string.ascii_lowercase
        elif case == "upper":
            alpha = string.ascii_uppercase
        elif case == "lower":
            alpha = string.ascii_lowercase
        else:
            raise ValueError("case can only be 'both', 'upper' or 'lower'")
    else:
        alpha = string.ascii_uppercase + string.ascii_lowercase
    if punctuation:
        limit_punctuation = kwargs.get('limit_punctuation', '')
        if limit_punctuation == '':
            punctuation_set = string.punctuation
        else:
            # In case limit_punctuation contains non-punctuation characters
            punctuation_set = ''.join([p for p in limit_punctuation
                                   if p in string.punctuation])
    else:
        punctuation_set = string.punctuation

    srandom = random.SystemRandom()
    p_generator = Generator(punctuation_set, srandom, p_min, p_max)
    d_generator = Generator(string.digits, srandom, d_min, d_max)
    a_generator = Generator(alpha, srandom, a_min, a_max)

    main_generator = SuperGenerator(srandom, length, length)
    main_generator.add(p_generator)
    main_generator.add(a_generator)
    main_generator.add(d_generator)
    chars = []
    for i in main_generator:
        chars.append(i)
    try:
        srandom.shuffle(chars, srandom)
    except:
        random.shuffle(chars)
    return "".join(chars)