def filtre(liste_base, criteres) -> groups.Collection:
        """
        Return a filter list, bases on criteres

        :param liste_base: Acces list
        :param criteres: Criteria { `attribut`:[valeurs,...] }
        """

        def choisi(ac):
            for cat, li in criteres.items():
                v = ac[cat]
                if not (v in li):
                    return False
            return True

        return groups.Collection(a for a in liste_base if choisi(a))