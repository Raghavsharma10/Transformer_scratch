def get_dbg_brk_linux64():
    '''
    Return the current brk value in the debugged process (only x86_64 Linux)
    '''
    # TODO this method is so weird, find a unused address to inject code not
    # the base address

    debugger = get_debugger()

    code = b'\x0f\x05'  # syscall

    rax = debugger.get_reg("rax")
    rdi = debugger.get_reg("rdi")
    rip = debugger.get_reg("rip")
    efl = debugger.get_reg("efl")

    debugger.set_reg("rax", 12)  # sys_brk
    debugger.set_reg("rdi", 0)

    base = debugger.image_base()

    inj = base

    save = debugger.get_bytes(inj, len(code))

    debugger.put_bytes(inj, code)

    debugger.set_reg("rip", inj)

    debugger.step_into()
    debugger.wait_ready()

    brk_res = debugger.get_reg("rax")

    debugger.set_reg("rax", rax)
    debugger.set_reg("rdi", rdi)
    debugger.set_reg("rip", rip)
    debugger.set_reg("efl", efl)

    debugger.put_bytes(inj, save)

    return brk_res