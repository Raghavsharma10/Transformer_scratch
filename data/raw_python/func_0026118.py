def eparOptionFactory(master, statusBar, param, defaultParam,
                      doScroll, fieldWidths,
                      plugIn=None, editedCallbackObj=None,
                      helpCallbackObj=None, mainGuiObj=None,
                      defaultsVerb="Default", bg=None, indent=False,
                      flagging=False, flaggedColor=None):

    """Return EparOption item of appropriate type for the parameter param"""

    # Allow passed-in overrides
    if plugIn is not None:
        eparOption = plugIn

    # If there is an enumerated list, regardless of datatype use EnumEparOption
    elif param.choice is not None:
        eparOption = EnumEparOption

    else:
        # Use String for types not in the dictionary
        eparOption = _eparOptionDict.get(param.type, StringEparOption)

    # Create it
    eo = eparOption(master, statusBar, param, defaultParam, doScroll,
                    fieldWidths, defaultsVerb, bg,
                    indent=indent, helpCallbackObj=helpCallbackObj,
                    mainGuiObj=mainGuiObj)
    eo.setEditedCallbackObj(editedCallbackObj)
    eo.setIsFlagging(flagging, False)
    if flaggedColor:
        eo.setFlaggedColor(flaggedColor)
    return eo