def _parse_chemical_equation(value):
    """
    Parse the chemical equation mini-language.

    See the docstring of `ChemicalEquation` for more.

    Parameters
    ----------
    value : `str`
        A string in chemical equation mini-language.

    Returns
    -------
    mapping
        A mapping in the format specified by the mini-language (see notes on
        `ChemicalEquation`).

    Examples
    --------
    >>> from pyrrole.core import _parse_chemical_equation
    >>> parsed = _parse_chemical_equation('4 A + 3 B <- 2 C + D')
    >>> parsed['arrow']
    '->'
    >>> parsed['products'][1]['species']
    'B'
    >>> parsed['reactants'][0]['coefficient']
    2

    """
    arrow = _pp.oneOf('-> <- <=>').setResultsName('arrow')
    species = _pp.Word(_pp.printables).setResultsName('species')
    coefficient = (_pp.Optional(_pp.Word(_pp.nums), default=1)
                   .setParseAction(_pp.tokenMap(int))
                   .setResultsName('coefficient'))
    group_ = _pp.Group(coefficient + _pp.Optional(_pp.Suppress('*')) + species)
    reactants = ((group_ + _pp.ZeroOrMore(_pp.Suppress('+') + group_))
                 .setResultsName('reactants'))
    products = ((group_ + _pp.ZeroOrMore(_pp.Suppress('+') + group_))
                .setResultsName('products'))

    grammar = reactants + arrow + products
    parsed = grammar.parseString(value).asDict()

    if parsed['arrow'] == '<-':
        parsed['reactants'], parsed['products'] \
            = parsed['products'], parsed['reactants']
        parsed['arrow'] = '->'

    return parsed