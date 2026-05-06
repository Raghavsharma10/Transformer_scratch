def get_section_from_condition(if_section, else_section, kwargs):
    """Returns section that should be used from given if/else sections by evaluating given
    condition.

    Args:
        if_section - section with if clause
        else_section - section that *may* be else clause (just next section after if_section,
                       this method will check if it really is else); possibly None if not present

    Returns:
        tuple (<0 or 1>, <True or False>, section), where
        - the first member says whether we're going to "if" section (0) or else section (1)
        - the second member says whether we should skip next section during further evaluation
          (True or False - either we have else to skip, or we don't)
        - the third member is the appropriate section to run or None if there is only "if"
          clause and condition evaluates to False
    """
    # check if else section is really else
    skip = True if else_section is not None and else_section[0] == 'else' else False
    if evaluate_expression(if_section[0][2:].strip(), kwargs)[0]:
        return (0, skip, if_section[1])
    else:
        return (1, skip, else_section[1]) if skip else (1, skip, None)