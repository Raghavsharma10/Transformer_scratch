def _special_value_maxLength(em, newValue=NOT_PROVIDED):
    '''
        _special_value_maxLength - Handle the special "maxLength" property

            @param em <AdvancedTag> - The tag element

            @param newValue - Default NOT_PROVIDED, if provided will use that value instead of the

                current .getAttribute value on the tag. This is because this method can be used for both validation
                 
                and getting/setting
    '''
    
    if newValue is NOT_PROVIDED:
        if not em.hasAttribute('maxlength'):
            return -1

        curValue = em.getAttribute('maxlength', '-1')

        # If we are accessing, the invalid default should be negative
        invalidDefault = -1
    else:
        curValue = newValue

        # If we are setting, we should raise an exception upon invalid value
        invalidDefault = IndexSizeErrorException

    return convertToIntRange(curValue, minValue=0, maxValue=None, emptyValue='0', invalidDefault=invalidDefault)