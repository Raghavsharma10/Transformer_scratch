def _cond_select_key_nonrecur(d,cond_match=None,**kwargs):
    '''
        d = {
            "ActiveArea":"50829", 
            "Artist":"315",                 
            "AsShotPreProfileMatrix":"50832",
            "AnalogBalance":"50727",          
            "AsShotICCProfile":"50831",       
            "AsShotProfileName":"50934",
            "AntiAliasStrength":"50738",      
            "AsShotNeutral":"50728",          
            "AsShotWhiteXY":"50729"
        }
        _cond_select_key_nonrecur(d,"An")
        _cond_select_key_nonrecur(d,"As")
        regex = re.compile("e$")
        _cond_select_key_nonrecur(d,regex)
    '''
    if('cond_func' in kwargs):
        cond_func = kwargs['cond_func']
    else:
        cond_func = _text_cond
    if('cond_func_args' in kwargs):
        cond_func_args = kwargs['cond_func_args']
    else:
        cond_func_args = []
    rslt = {}
    for key in d:
        if(cond_func(key,cond_match,*cond_func_args)):
            rslt[key] = d[key]
        else:
            pass
    return(rslt)