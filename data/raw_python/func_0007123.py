def date_decoder(dic):
    """Add python types decoding. See JsonEncoder"""
    if '__date__' in dic:
        try:
            d = datetime.date(**{c: v for c, v in dic.items() if not c == "__date__"})
        except (TypeError, ValueError):
            raise json.JSONDecodeError("Corrupted date format !", str(dic), 1)
    elif '__datetime__' in dic:
        try:
            d = datetime.datetime(**{c: v for c, v in dic.items() if not c == "__datetime__"})
        except (TypeError, ValueError):
            raise json.JSONDecodeError("Corrupted datetime format !", str(dic), 1)
    else:
        return dic
    return d