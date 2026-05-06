def init(scope):
    """
    Initialize the xqt system with the PyQt4 wrapper for the Qt system.
    
    :param      scope | <dict>
    """
    # update globals
    scope['py2q'] = py2q
    scope['q2py'] = q2py
    
    # define wrapper compatibility symbols
    QtCore.THREADSAFE_NONE = None
    
    # define the importable symbols
    scope['QtCore'] = QtCore
    scope['QtGui'] = lazy_import('PyQt4.QtGui')
    scope['QtWebKit'] = lazy_import('PyQt4.QtWebKit')
    scope['QtNetwork'] = lazy_import('PyQt4.QtNetwork')
    scope['QtXml'] = lazy_import('PyQt4.QtXml')
    
    # PyQt4 specific modules
    scope['QtDesigner'] = lazy_import('PyQt4.QtDesigner')
    scope['Qsci'] = lazy_import('PyQt4.Qsci')
    
    scope['uic'] = lazy_import('PyQt4.uic')
    scope['rcc_exe'] = 'pyrcc4'
    
    # map shared core properties
    QtCore.QDate.toPython = lambda x: x.toPyDate()
    QtCore.QDateTime.toPython = lambda x: x.toPyDateTime()
    QtCore.QTime.toPython = lambda x: x.toPyTime()
    
    QtCore.Signal = Signal
    QtCore.Slot = Slot
    QtCore.Property = QtCore.pyqtProperty
    QtCore.SIGNAL = SIGNAL
    QtCore.__version__ = QtCore.QT_VERSION_STR

    if SIP_VERSION == '2':
        QtCore.QStringList = list
        QtCore.QString = unicode