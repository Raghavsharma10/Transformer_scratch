def _special_value_size(em):
    '''
        handle "size" property, which has different behaviour for input vs everything else
    '''
    if em.tagName == 'input':
        # TODO: "size" on an input is implemented very weirdly. Negative values are treated as invalid,
        #          A value of "0" raises an exception (and does not set HTML attribute)
        #          No upper limit.
        return convertToPositiveInt(em.getAttribute('size', 20), invalidDefault=20)
    return em.getAttribute('size', '')