def apply_string_substitutions(
    inputs,
    substitutions,
    inverse=False,
    case_insensitive=False,
    unused_substitutions="ignore",
):
    """Apply a number of substitutions to a string(s).

    The substitutions are applied effectively all at once. This means that conflicting
    substitutions don't interact. Where substitutions are conflicting, the one which
    is longer takes precedance. This is confusing so we recommend that you look at
    the examples.

    Parameters
    ----------
    inputs : str, list of str
        The string(s) to which we want to apply the substitutions.

    substitutions : dict
        The substitutions we wish to make. The keys are the strings we wish to
        substitute, the values are the strings which we want to appear in the output
        strings.

    inverse : bool
        If True, do the substitutions the other way around i.e. use the keys as the
        strings we want to appear in the output strings and the values as the strings
        we wish to substitute.

    case_insensitive : bool
        If True, the substitutions will be made in a case insensitive way.

    unused_substitutions : {"ignore", "warn", "raise"}, default ignore
        Behaviour when one or more of the inputs does not have a corresponding
        substitution. If "ignore", nothing happens. If "warn", a warning is issued. If
        "raise", an error is raised. See the examples.

    Returns
    -------
    ``type(input)``
        The input with substitutions performed.

    Examples
    --------
    >>> apply_string_substitutions("Hello JimBob", {"Jim": "Bob"})
    'Hello BobBob'

    >>> apply_string_substitutions("Hello JimBob", {"Jim": "Bob"}, inverse=True)
    'Hello JimJim'

    >>> apply_string_substitutions(["Hello JimBob", "Jim says, 'Hi Bob'"], {"Jim": "Bob"})
    ['Hello BobBob', "Bob says, 'Hi Bob'"]

    >>> apply_string_substitutions(["Hello JimBob", "Jim says, 'Hi Bob'"], {"Jim": "Bob"}, inverse=True)
    ['Hello JimJim', "Jim says, 'Hi Jim'"]

    >>> apply_string_substitutions("Muttons Butter", {"M": "B", "Button": "Zip"})
    'Buttons Butter'
    # Substitutions don't cascade. If they did, Muttons would become Buttons, then the
    # substitutions "Button" --> "Zip" would be applied and we would end up with
    # "Zips Butter".

    >>> apply_string_substitutions("Muttons Butter", {"Mutton": "Gutter", "tt": "zz"})
    'Gutters Buzzer'
    # Longer substitutions take precedent. Hence Mutton becomes Gutter, not Muzzon.

    >>> apply_string_substitutions("Butter", {"buTTer": "Gutter"}, case_insensitive=True)
    'Gutter'

    >>> apply_string_substitutions("Butter", {"teeth": "tooth"})
    'Butter'

    >>> apply_string_substitutions("Butter", {"teeth": "tooth"}, unused_substitutions="ignore")
    'Butter'

    >>> apply_string_substitutions("Butter", {"teeth": "tooth"}, unused_substitutions="warn")
    ...pymagicc/utils.py:50: UserWarning: No substitution available for {'Butter'} warnings.warn(msg)
    'Butter'

    >>> apply_string_substitutions("Butter", {"teeth": "tooth"}, unused_substitutions="raise")
    ValueError: No substitution available for {'Butter'}
    """
    if inverse:
        substitutions = {v: k for k, v in substitutions.items()}

    # only possible to have conflicting substitutions when case insensitive
    if case_insensitive:
        _check_duplicate_substitutions(substitutions)

    if unused_substitutions != "ignore":
        _check_unused_substitutions(
            substitutions, inputs, unused_substitutions, case_insensitive
        )

    compiled_regexp = _compile_replacement_regexp(
        substitutions, case_insensitive=case_insensitive
    )

    inputs_return = deepcopy(inputs)
    if isinstance(inputs_return, str):
        inputs_return = _multiple_replace(inputs_return, substitutions, compiled_regexp)
    else:
        inputs_return = [
            _multiple_replace(v, substitutions, compiled_regexp) for v in inputs_return
        ]

    return inputs_return