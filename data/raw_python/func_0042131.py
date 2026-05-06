def _normalize_dakuten(self, chars):
        '''
        Replaces the dakuten and handakuten modifier character combinations
        with single characters. For example, か\u3099か becomes がけ,
        or は゜は becomes ぱは.
        '''
        prev = None
        prev_n = None

        # Set all repeater characters to 0 initially,
        # then go through the list and remove them all.
        for n in range(len(chars)):
            char = chars[n]

            if char in dkt:
                chars[n] = 0
                if prev in dkt_cvs:
                    chars[prev_n] = dkt_lt[prev]

            if char in hdkt:
                chars[n] = 0
                if prev in hdkt_cvs:
                    chars[prev_n] = hdkt_lt[prev]

            prev = char
            prev_n = n

        # Remove all 0 values. There should not be any other than the ones we
        # just added. (This could use (0).__ne__, but that's Python 3 only.)
        return list(filter(lambda x: x is not 0, chars))