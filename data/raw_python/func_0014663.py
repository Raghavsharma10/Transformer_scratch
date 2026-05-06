def toggleAttributesDOM(isEnabled):
    '''
        toggleAttributesDOM - Toggle if the old DOM tag.attributes NamedNodeMap model should be used for the .attributes method, versus

           a more sane direct dict implementation.

            The DOM version is always accessable as AdvancedTag.attributesDOM
            The dict version is always accessable as AdvancedTag.attributesDict

            Default for AdvancedTag.attributes is to be attributesDict implementation.

          @param isEnabled <bool> - If True, .attributes will be changed to use the DOM-provider. Otherwise, it will use the dict provider.
    '''

    if isEnabled:
        AdvancedTag.attributes = AdvancedTag.attributesDOM
    else:
        AdvancedTag.attributes = AdvancedTag.attributesDict