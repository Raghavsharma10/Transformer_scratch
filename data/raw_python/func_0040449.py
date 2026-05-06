def createActionGroup(self, parent=None, name=''):
        """
        Overloads teh create action method to handle the proper base
        instance information, similar to the PyQt4 loading system.
        
        :param      parent | <QWidget> || None
                    name   | <str>
        """
        actionGroup = super(UiLoader, self).createActionGroup(parent, name)
        if not actionGroup.parent():
            actionGroup.setParent(self._baseinstance)
        setattr(self._baseinstance, name, actionGroup)
        return actionGroup