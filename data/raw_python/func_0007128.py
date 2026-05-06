def in_dateheure(objet, pattern):
        """ abstractSearch dans une date-heure datetime.datetime (cf abstractRender.dateheure) """
        if objet:
            pattern = re.sub(" ", '', pattern)
            objet_str = abstractRender.dateheure(objet)
            return bool(re.search(pattern, objet_str))
        return False