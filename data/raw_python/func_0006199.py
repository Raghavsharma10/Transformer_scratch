def dmstodd(self, dms):
        """ convert dms to dd"""
        size = len(dms)
        letters = 'WENS'
        is_annotated = False

        try:
            float(dms)
        except ValueError:
            for letter in letters:
                if letter in dms.upper():
                    is_annotated = True
                    break
            if not is_annotated:
                raise core.RTreeError("unable to parse '%s' to decimal degrees" % dms)
        is_negative = False
        if is_annotated:
            dms_upper = dms.upper()
            if 'W' in dms_upper or 'S' in dms_upper:
                is_negative = True
        else:
            if dms < 0:
                is_negative = True

        if is_annotated:
            bletters = letters.encode(encoding='utf-8')
            bdms = dms.encode(encoding = 'utf-8')
            dms = bdms.translate(None, bletters).decode('ascii')

            # bletters = bytes(letters, encoding='utf-8')
            # bdms = bytes(dms, encoding='utf-8')
            # dms = bdms.translate(None, bletters).decode('ascii')

            # dms = dms.translate(None, letters) # Python 2.x version

        pieces = dms.split(".")
        D = 0.0
        M = 0.0
        S = 0.0
        divisor = 3600.0
        if len(pieces) == 1:
            S = dms[-2:]
            M = dms[-4:-2]
            D = dms[:-4]
        else:
            S = '{0:s}.{1:s}'.format (pieces[0][-2:], pieces[1])
            M = pieces[0][-4:-2]
            D = pieces[0][:-4]

        DD = float(D) + float(M)/60.0 + float(S)/divisor
        if is_negative:
            DD = DD * -1.0
        return DD