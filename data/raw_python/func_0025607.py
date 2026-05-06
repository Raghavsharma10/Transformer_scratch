def quick_summary(nml2_doc):
    '''
    Or better just use nml2_doc.summary(show_includes=False)
    '''
    
    info = 'Contents of NeuroML 2 document: %s\n'%nml2_doc.id
    membs = inspect.getmembers(nml2_doc)

    for memb in membs:

        if isinstance(memb[1], list) and len(memb[1])>0 \
                and not memb[0].endswith('_'):
            info+='  %s:\n    ['%memb[0]
            for entry in memb[1]:
                extra = '???'
                extra = entry.name if hasattr(entry,'name') else extra
                extra = entry.href if hasattr(entry,'href') else extra
                extra = entry.id if hasattr(entry,'id') else extra
                
                info+=" %s (%s),"%(entry, extra)
            
            info+=']\n'
    return info