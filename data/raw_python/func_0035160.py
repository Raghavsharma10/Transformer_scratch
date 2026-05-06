def _my_pdf_formatter(data, format, ordered_alphabets) :
    """ Generate a logo in PDF format.
    
    Modified from weblogo version 3.4 source code.
    """
    eps = _my_eps_formatter(data, format, ordered_alphabets).decode()
    gs = weblogolib.GhostscriptAPI()    
    return gs.convert('pdf', eps, format.logo_width, format.logo_height)