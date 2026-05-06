def teal(theTask, parent=None, loadOnly=False, returnAs="dict",
         canExecute=True, strict=False, errorsToTerm=False,
         autoClose=True, defaults=False):
#        overrides=None):
    """ Start the GUI session, or simply load a task's ConfigObj. """
    if loadOnly: # this forces returnAs="dict"
        obj = None
        try:
            obj = cfgpars.getObjectFromTaskArg(theTask, strict, defaults)
#           obj.strictUpdate(overrides) # ! would need to re-verify after this !
        except Exception as re: # catches RuntimeError and KeyError and ...
            # Since we are loadOnly, don't pop up the GUI for this
            if strict:
                raise
            else:
                print(re.message.replace('\n\n','\n'))
        return obj
    else:
        assert returnAs in ("dict", "status", None), \
               "Invalid value for returnAs arg: "+str(returnAs)
        dlg = None
        try:
            # if setting to all defaults, go ahead and load it here, pre-GUI
            if defaults:
                theTask = cfgpars.getObjectFromTaskArg(theTask, strict, True)
            # now create/run the dialog
            dlg = ConfigObjEparDialog(theTask, parent=parent,
                                      autoClose=autoClose,
                                      strict=strict,
                                      canExecute=canExecute)
#                                     overrides=overrides)
        except cfgpars.NoCfgFileError as ncf:
            log_last_error()
            if errorsToTerm:
                print(str(ncf).replace('\n\n','\n'))
            else:
                popUpErr(parent=parent,message=str(ncf),title="Unfound Task")
        except Exception as re: # catches RuntimeError and KeyError and ...
            log_last_error()
            if errorsToTerm:
                print(re.message.replace('\n\n','\n'))
            else:
                popUpErr(parent=parent, message=re.message,
                         title="Bad Parameters")

        # Return, depending on the mode in which we are operating
        if returnAs is None:
            return

        if returnAs == "dict":
            if dlg is None or dlg.canceled():
                return None
            else:
                return dlg.getTaskParsObj()

        # else, returnAs == "status"
        if dlg is None or dlg.canceled():
            return -1
        if dlg.executed():
            return 1
        return 0