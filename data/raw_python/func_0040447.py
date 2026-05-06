def init(scope):
    """
    Initialize the xqt system with the PySide wrapper for the Qt system.
    
    :param      scope | <dict>
    """
    # define wrapper compatibility symbols
    QtCore.THREADSAFE_NONE = XThreadNone()
    QtGui.QDialog = QDialog
    
    # define the importable symbols
    scope['QtCore'] = QtCore
    scope['QtGui'] = QtGui
    scope['QtWebKit'] = lazy_import('PySide.QtWebKit')
    scope['QtNetwork'] = lazy_import('PySide.QtNetwork')
    scope['QtXml'] = lazy_import('PySide.QtXml')
    
    scope['uic'] = Uic()
    scope['rcc_exe'] = 'pyside-rcc'
    
    # map overrides
    #QtCore.SIGNAL = SIGNAL
    
    # map shared core properties
    QtCore.QDate.toPyDate = lambda x: x.toPython()
    QtCore.QDateTime.toPyDateTime = lambda x: x.toPython()
    QtCore.QTime.toPyTime = lambda x: x.toPython()
    QtCore.QStringList = list
    QtCore.QString = unicode