def createLayout(self, className, parent=None, name=''):
        """
        Overloads teh create action method to handle the proper base
        instance information, similar to the PyQt4 loading system.
        
        :param      className | <str>
                    parent | <QWidget> || None
                    name   | <str>
        """
        layout = super(UiLoader, self).createLayout(className, parent, name)
        setattr(self._baseinstance, name, layout)
        return layout