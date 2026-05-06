def _special_value_rows(em):
    '''
        _special_value_rows - Handle "rows" special attribute, which differs if tagName is a textarea or frameset
    '''
    if em.tagName == 'textarea':
        return convertToIntRange(em.getAttribute('rows', 2), minValue=1, maxValue=None, invalidDefault=2)
    else:
        # frameset
        return em.getAttribute('rows', '')