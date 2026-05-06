def exceptionToString(e,silent=False):
    """when you "except Exception as e", give me the e and I'll give you a string."""
    exc_type, exc_obj, exc_tb = sys.exc_info()
    s=("\n"+"="*50+"\n")
    s+="EXCEPTION THROWN UNEXPECTEDLY\n"
    s+="  FILE: %s\n"%os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    s+="  LINE: %s\n"%exc_tb.tb_lineno
    s+="  TYPE: %s\n"%exc_type
    s+='-'*50+'\n'
    s+=traceback.format_exc()
    s=s.strip()+'\n'+"="*50+"\n"
    if not silent:
        print(s)
    return s