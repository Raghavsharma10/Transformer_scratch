def jsonise(dic):
        """Renvoie un dictionnaire dont les champs dont compatibles avec SQL
        Utilise Json. Attention à None : il faut laisser None et non pas null"""
        d = {}
        for k, v in dic.items():
            if type(v) in abstractRequetesSQL.TYPES_PERMIS:
                d[k] = v
            else:
                try:
                    d[k] = json.dumps(v, ensure_ascii=False, cls=formats.JsonEncoder)
                except ValueError as e:
                    logging.exception("Erreur d'encodage JSON !")
                    raise e
        return d