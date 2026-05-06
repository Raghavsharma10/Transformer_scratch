def _nuke_set_zero_margins(widget_object):
    """Remove Nuke margins when docked UI
    .. _More info:
        https://gist.github.com/maty974/4739917
    """
    parentApp = QtWidgets.QApplication.allWidgets()
    parentWidgetList = []
    for parent in parentApp:
        for child in parent.children():
            if widget_object.__class__.__name__ == child.__class__.__name__:
                parentWidgetList.append(
                    parent.parentWidget())
                parentWidgetList.append(
                    parent.parentWidget().parentWidget())
                parentWidgetList.append(
                    parent.parentWidget().parentWidget().parentWidget())

                for sub in parentWidgetList:
                        for tinychild in sub.children():
                            try:
                                tinychild.setContentsMargins(0, 0, 0, 0)
                            except Exception:
                                pass