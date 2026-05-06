def _legacySpecialCases(form, patterns, parameter):
    """
    Create a view object for the given parameter.

    This function implements the remaining view construction logic which has
    not yet been converted to the C{viewFactory}-style expressed in
    L{_LiveFormMixin.form}.

    @type form: L{_LiveFormMixin}
    @param form: The form fragment which contains the given parameter.
    @type patterns: L{PatternDictionary}
    @type parameter: L{Parameter}, L{ChoiceParameter}, or L{ListParameter}.
    """
    p = patterns[parameter.type + '-input-container']

    if parameter.type == TEXTAREA_INPUT:
        p = dictFillSlots(p, dict(label=parameter.label,
                                  name=parameter.name,
                                  value=parameter.default or ''))
    elif parameter.type == MULTI_TEXT_INPUT:
        subInputs = list()

        for i in xrange(parameter.count):
            subInputs.append(dictFillSlots(patterns['input'],
                                dict(name=parameter.name + '_' + str(i),
                                     type='text',
                                     value=parameter.defaults[i])))

        p = dictFillSlots(p, dict(label=parameter.label or parameter.name,
                                  inputs=subInputs))

    else:
        if parameter.default is not None:
            value = parameter.default
        else:
            value = ''

        if parameter.type == CHECKBOX_INPUT and parameter.default:
            inputPattern = 'checked-checkbox-input'
        else:
            inputPattern = 'input'

        p = dictFillSlots(
            p, dict(label=parameter.label or parameter.name,
                    input=dictFillSlots(patterns[inputPattern],
                                        dict(name=parameter.name,
                                             type=parameter.type,
                                             value=value))))

    p(**{'class' : 'liveform_'+parameter.name})

    if parameter.description:
        description = patterns['description'].fillSlots(
                           'description', parameter.description)
    else:
        description = ''

    return dictFillSlots(
        patterns['parameter-input'],
        dict(input=p, description=description))