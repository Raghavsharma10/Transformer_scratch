def in_telephones(objet, pattern):
        """ abstractSearch dans une liste de téléphones."""
        objet = objet or []
        if pattern == '' or not objet:
            return False
        return max(bool(re.search(pattern, t)) for t in objet)