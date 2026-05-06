def dateheure(objet):
        """ abstractRender d'une date-heure datetime.datetime au format JJ/MM/AAAAàHH:mm """
        if objet:
            return "{}/{}/{} à {:02}:{:02}".format(objet.day, objet.month, objet.year, objet.hour, objet.minute)
        return ""