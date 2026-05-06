def fixLabel(label, maxlen, delim=None, repl='', truncend=True):
    """Truncate long graph and field labels.
    
        @param label:    Label text.
        @param maxlen:   Maximum field label length in characters.
                         No maximum field label length is enforced by default.
        @param delim:    Delimiter for field labels field labels longer than 
                         maxlen will preferably be truncated at delimiter.
        @param repl:     Replacement string for truncated part.
        @param truncend: Truncate the end of label name if True. (Default)
                         The beginning part of label will be truncated if False.
                         
    """
    if len(label) <= maxlen:
        return label
    else:
        maxlen -= len(repl)
        if delim is not None:  
            if truncend:
                end = label.rfind(delim, 0, maxlen)
                if end > 0:
                    return label[:end+1] + repl
            else:
                start = label.find(delim, len(label) - maxlen)
                if start > 0:
                    return repl + label[start:]
        if truncend:
            return label[:maxlen] + repl
        else:
            return repl + label[-maxlen:]