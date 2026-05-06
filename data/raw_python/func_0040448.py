def createAction(self, parent=None, name=''):
        """
        Overloads teh create action method to handle the proper base
        instance information, similar to the PyQt4 loading system.
        
        :param      parent | <QWidget> || None
                    name   | <str>
        """
        action = super(UiLoader, self).createAction(parent, name)
        if not action.parent():
            action.setParent(self._baseinstance)
        setattr(self._baseinstance, name, action)
        return action