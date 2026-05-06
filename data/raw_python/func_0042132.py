def _process_repeaters(self, chars):
        '''
        Replace all repeater characters (e.g. turn サヾエ into サザエ).
        '''
        prev = None
        for n in range(len(chars)):
            char = chars[n]
            if char in rpts:
                # The character is a repeater.
                chars[n] = prev

            if char in drpts:
                # The character is a repeater with dakuten.
                # If the previous character can have a dakuten, add that
                # to the stack; if not, just add whatever we had previously.
                if prev in dkt_cvs:
                    chars[n] = dkt_lt[prev]
                else:
                    chars[n] = prev

            prev = char

        return chars