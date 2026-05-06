def _special_value_autocomplete(em):
    '''
        handle "autocomplete" property, which has different behaviour for form vs input"
    '''
    if em.tagName == 'form':
        return convertPossibleValues(em.getAttribute('autocomplete', 'on'), POSSIBLE_VALUES_ON_OFF, invalidDefault='on', emptyValue=EMPTY_IS_INVALID)
    # else: input
    return convertPossibleValues(em.getAttribute('autocomplete', ''), POSSIBLE_VALUES_ON_OFF, invalidDefault="", emptyValue='')