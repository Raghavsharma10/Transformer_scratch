def get_dbg_brk_linux32():
    '''
    Return the current brk value in the debugged process (only x86 Linux)
    '''
    # TODO this method is so weird, find a unused address to inject code not
    # the base address

    debugger = get_debugger()

    code = b'\xcd\x80'  # int 0x80

    eax = debugger.get_reg("eax")
    ebx = debugger.get_reg("ebx")
    eip = debugger.get_reg("eip")
    efl = debugger.get_reg("efl")

    debugger.set_reg("eax", 45)  # sys_brk
    debugger.set_reg("ebx", 0)

    base = debugger.image_base()

    inj = base

    save = debugger.get_bytes(inj, len(code))

    debugger.put_bytes(inj, code)

    debugger.set_reg("eip", inj)

    debugger.step_into()
    debugger.wait_ready()

    brk_res = debugger.get_reg("eax")

    debugger.set_reg("eax", eax)
    debugger.set_reg("ebx", ebx)
    debugger.set_reg("eip", eip)
    debugger.set_reg("efl", efl)

    debugger.put_bytes(inj, save)

    return brk_res