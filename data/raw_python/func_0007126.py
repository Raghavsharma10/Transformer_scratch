def in_string(objet, pattern):
        """ abstractSearch dans une chaine, sans tenir compte de la casse. """
        return bool(re.search(pattern, str(objet), flags=re.I)) if objet else False