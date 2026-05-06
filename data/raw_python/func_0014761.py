def _special_value_cols(em):
    '''
        _special_value_cols - Handle "cols" special attribute, which differs if tagName is a textarea or frameset
    '''
    if em.tagName == 'textarea':
        return convertToIntRange(em.getAttribute('cols', 20), minValue=1, maxValue=None, invalidDefault=20)
    else:
        # frameset
        return em.getAttribute('cols', '')