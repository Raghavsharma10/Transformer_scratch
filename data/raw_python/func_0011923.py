def titleify(self, lang='en', allwords=False, lastword=True):
        """takes a string and makes a title from it"""
        if lang in LOWERCASE_WORDS:
            lc_words = LOWERCASE_WORDS[lang]
        else:
            lc_words = []
        s = str(self).strip()
        l = re.split(r"([_\W]+)", s)
        for i in range(len(l)):
            l[i] = l[i].lower()
            if (
                allwords == True
                or i == 0
                or (lastword == True and i == len(l) - 1)
                or l[i].lower() not in lc_words
            ):
                w = l[i]
                if len(w) > 1:
                    w = w[0].upper() + w[1:]
                else:
                    w = w.upper()
                l[i] = w
        s = "".join(l)
        return String(s)