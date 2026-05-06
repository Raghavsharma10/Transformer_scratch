def showMessageDialog(title, text):
    '''
    Show a dialog containing a given text, with a given title.

    The text accepts HTML syntax
    '''
    dlg = QgsMessageOutput.createMessageOutput()
    dlg.setTitle(title)
    dlg.setMessage(text, QgsMessageOutput.MessageHtml)
    dlg.showMessage()