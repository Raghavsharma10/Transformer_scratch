def execute(func, message = None):
    '''
    Executes a lengthy tasks in a separate thread and displays a waiting dialog if needed.
    Sets the cursor to wait cursor while the task is running.

    This function does not provide any support for progress indication

    :param func: The function to execute.

    :param message: The message to display in the wait dialog. If not passed, the dialog won't be shown
    '''
    global _dialog
    cursor = QtWidgets.QApplication.overrideCursor()
    waitCursor = (cursor is not None and cursor.shape() == QtCore.Qt.WaitCursor)
    dialogCreated = False
    try:
        QtCore.QCoreApplication.processEvents()
        if not waitCursor:
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        if message is not None:
            t = ExecutorThread(func)
            loop = QtCore.QEventLoop()
            t.finished.connect(loop.exit, QtCore.Qt.QueuedConnection)
            if _dialog is None:
                dialogCreated = True
                _dialog = QtGui.QProgressDialog(message, "Running", 0, 0, iface.mainWindow())
                _dialog.setWindowTitle("Running")
                _dialog.setWindowModality(QtCore.Qt.WindowModal);
                _dialog.setMinimumDuration(1000)
                _dialog.setMaximum(100)
                _dialog.setValue(0)
                _dialog.setMaximum(0)
                _dialog.setCancelButton(None)
            else:
                oldText = _dialog.labelText()
                _dialog.setLabelText(message)
            QtWidgets.QApplication.processEvents()
            t.start()
            loop.exec_(flags = QtCore.QEventLoop.ExcludeUserInputEvents)
            if t.exception is not None:
                raise t.exception
            return t.returnValue
        else:
            return func()
    finally:
        if message is not None:
            if dialogCreated:
                _dialog.reset()
                _dialog = None
            else:
                _dialog.setLabelText(oldText)
        if not waitCursor:
            QtWidgets.QApplication.restoreOverrideCursor()
        QtCore.QCoreApplication.processEvents()