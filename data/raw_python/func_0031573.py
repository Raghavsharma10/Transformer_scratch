def inst_repr(instance, fmt='str', public_only=True):
    """
    Generate class instance signature from its __dict__
    From python 3.6 dict is ordered and order of attributes will be preserved automatically

    Args:
        instance: class instance
        fmt: ['json', 'str']
        public_only: if display public members only

    Returns:
        str: string or json representation of instance

    Examples:
        >>> inst_repr(1)
        ''
        >>> class SampleClass(object):
        ...     def __init__(self):
        ...         self.b = 3
        ...         self.a = 4
        ...         self._private_ = 'hidden'
        >>>
        >>> s = SampleClass()
        >>> inst_repr(s)
        '{b=3, a=4}'
        >>> inst_repr(s, public_only=False)
        '{b=3, a=4, _private_=hidden}'
        >>> json.loads(inst_repr(s, fmt='json'))
        {'b': 3, 'a': 4}
        >>> inst_repr(s, fmt='unknown')
        ''
    """
    if not hasattr(instance, '__dict__'): return ''

    if public_only: inst_dict = {k: v for k, v in instance.__dict__.items() if k[0] != '_'}
    else: inst_dict = instance.__dict__

    if fmt == 'json': return json.dumps(inst_dict, indent=2)
    elif fmt == 'str': return to_str(inst_dict, public_only=public_only)

    return ''