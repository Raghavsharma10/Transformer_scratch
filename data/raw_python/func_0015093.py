def remote_debug(sig,frame):
    """Handler to allow process to be remotely debugged."""
    def _raiseEx(ex):
        """Raise specified exception in the remote process"""
        _raiseEx.ex = ex
    _raiseEx.ex = None
    
    try:
        # Provide some useful functions.
        locs = {'_raiseEx' : _raiseEx}
        locs.update(frame.f_locals)  # Unless shadowed.
        globs = frame.f_globals
        
        pid = os.getpid()  # Use pipe name based on pid
        pipe = NamedPipe(pipename(pid))
    
        old_stdout, old_stderr = sys.stdout, sys.stderr
        txt = ''
        pipe.put("Interrupting process at following point:\n" + 
               ''.join(traceback.format_stack(frame)) + ">>> ")
        
        try:
            while pipe.is_open() and _raiseEx.ex is None:
                line = pipe.get()
                if line is None: continue # EOF
                txt += line
                try:
                    code = codeop.compile_command(txt)
                    if code:
                        sys.stdout = six.StringIO()
                        sys.stderr = sys.stdout
                        six.exec_(code, globs, locs)
                        txt = ''
                        pipe.put(sys.stdout.getvalue() + '>>> ')
                    else:
                        pipe.put('... ')
                except:
                    txt='' # May be syntax err.
                    sys.stdout = six.StringIO()
                    sys.stderr = sys.stdout
                    traceback.print_exc()
                    pipe.put(sys.stdout.getvalue() + '>>> ')
        finally:
            sys.stdout = old_stdout # Restore redirected output.
            sys.stderr = old_stderr
            pipe.close()

    except Exception:  # Don't allow debug exceptions to propogate to real program.
        traceback.print_exc()
        
    if _raiseEx.ex is not None: raise _raiseEx.ex