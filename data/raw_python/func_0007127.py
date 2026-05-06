def in_date(objet, pattern):
        """ abstractSearch dans une date datetime.date"""
        if objet:
            pattern = re.sub(" ", '', pattern)
            objet_str = abstractRender.date(objet)
            return bool(re.search(pattern, objet_str))
        return False