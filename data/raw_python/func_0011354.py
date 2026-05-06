def int3(params, ctxt, scope, stream, coord, interp):
    """Define the ``Int3()`` function in the interpreter. Calling
    ``Int3()`` will drop the user into an interactive debugger.
    """
    if interp._no_debug:
        return

    if interp._int3:
        interp.debugger = PfpDbg(interp)
        interp.debugger.cmdloop()