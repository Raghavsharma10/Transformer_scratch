def createWidget(self, className, parent=None, name=''):
        """
        Overloads the createWidget method to handle the proper base instance
        information similar to the PyQt4 loading system.
        
        :param      className | <str>
                    parent    | <QWidget> || None
                    name      | <str>
        
        :return     <QWidget>
        """
        className = str(className)
        
        # create a widget off one of our dynamic classes
        if className in self.dynamicWidgets:
            widget = self.dynamicWidgets[className](parent)
            if parent:
                widget.setPalette(parent.palette())
            widget.setObjectName(name)
            
            # hack fix on a QWebView (will crash app otherwise)
            # forces a URL to the QWebView before it finishes
            if className == 'QWebView':
                widget.setUrl(QtCore.QUrl('http://www.google.com'))
        
        # create a widget from the default system
        else:
            widget = super(UiLoader, self).createWidget(className, parent, name)
            if parent:
                widget.setPalette(parent.palette())
        
        if parent is None:
            return self._baseinstance
        else:
            setattr(self._baseinstance, name, widget)
            return widget