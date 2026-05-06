def date(objet):
        """ abstractRender d'une date datetime.date"""
        if objet:
            return "{}/{}/{}".format(objet.day, objet.month, objet.year)
        return ""