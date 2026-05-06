def recherche(self, pattern, entete):
        """Performs a search field by field, using functions defined in formats.
        Matchs are marked with info[`font`]

        :param pattern: String to look for
        :param entete: Fields to look into
        :return: Nothing. The collection is changed in place
        """

        new_liste = []
        sub_patterns = pattern.split(" ")
        for p in self:
            d_font = {att: False for att in entete}
            row_valid = True
            for sub_pattern in sub_patterns:
                found = False
                for att in entete:
                    fonction_recherche = formats.ASSOCIATION[att][1]
                    attr_found = bool(fonction_recherche(p[att], sub_pattern))
                    if attr_found:
                        found = True
                        d_font[att] = True
                if not found:
                    row_valid = False
                    break
            if row_valid:
                new_liste.append(p)
                info = dict(self.get_info(Id=p.Id),font=d_font)
                self.infos[p.Id] = info

        list.__init__(self, new_liste)