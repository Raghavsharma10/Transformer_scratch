def __fix_args(kwargs):
    """
    Set all named arguments shortcuts and flags.
    """
    kwargs.setdefault('fixed_strings', kwargs.get('F'))
    kwargs.setdefault('basic_regexp', kwargs.get('G'))
    kwargs.setdefault('extended_regexp', kwargs.get('E'))
    kwargs.setdefault('ignore_case', kwargs.get('i'))
    kwargs.setdefault('invert', kwargs.get('v'))
    kwargs.setdefault('words', kwargs.get('w'))
    kwargs.setdefault('line', kwargs.get('x'))
    kwargs.setdefault('count', kwargs.get('c'))
    kwargs.setdefault('max_count', kwargs.get('m'))
    kwargs.setdefault('after_context', kwargs.get('A'))
    kwargs.setdefault('before_context', kwargs.get('B'))
    kwargs.setdefault('quiet', kwargs.get('q'))
    kwargs.setdefault('byte_offset', kwargs.get('b'))
    kwargs.setdefault('only_matching', kwargs.get('o'))
    kwargs.setdefault('line_number', kwargs.get('n'))
    kwargs.setdefault('regex_flags', kwargs.get('r'))
    kwargs.setdefault('keep_eol', kwargs.get('k'))
    kwargs.setdefault('trim', kwargs.get('t'))