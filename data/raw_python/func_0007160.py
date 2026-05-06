def sort(self, attribut, order=False):
        """
        Implément un tri par attrbut.

        :param str attribut: Nom du champ concerné
        :param bool order: Ordre croissant ou décroissant
        """
        value_default = formats.ASSOCIATION[attribut][3]

        if type(value_default) is str:  # case insensitive sort
            get = lambda d : (d[attribut] or value_default).casefold()
        elif type(value_default) is dict: #can't sort dicts
            def get(d):
                u = d[attribut] or value_default
                return [str(u[i]) for i in sorted(u.keys())]
        else:
            get = lambda d : d[attribut] or value_default

        list.sort(self, key=get, reverse=order)