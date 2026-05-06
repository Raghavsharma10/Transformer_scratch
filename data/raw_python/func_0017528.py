def __put_buttons_in_buttonframe(choices):
    """Put the buttons in the buttons frame"""
    global __widgetTexts, __firstWidget, buttonsFrame

    __firstWidget = None
    __widgetTexts = {}

    i = 0

    for buttonText in choices:
        tempButton = tk.Button(buttonsFrame, takefocus=1, text=buttonText)
        _bindArrows(tempButton)
        tempButton.pack(expand=tk.YES, side=tk.LEFT, padx='1m', pady='1m', ipadx='2m', ipady='1m')

        # remember the text associated with this widget
        __widgetTexts[tempButton] = buttonText

        # remember the first widget, so we can put the focus there
        if i == 0:
            __firstWidget = tempButton
            i = 1

        # for the commandButton, bind activation events to the activation event handler
        commandButton  = tempButton
        handler = __buttonEvent
        for selectionEvent in STANDARD_SELECTION_EVENTS:
            commandButton.bind('<%s>' % selectionEvent, handler)

        if CANCEL_TEXT in choices:
            commandButton.bind('<Escape>', __cancelButtonEvent)