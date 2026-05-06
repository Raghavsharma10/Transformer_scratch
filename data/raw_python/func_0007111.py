def load_from_local(cls):
        """Load datas from local file."""
        try:
            with open(cls.LOCAL_DB_PATH, 'rb') as f:
                b = f.read()
                s = security.protege_data(b, False)
        except (FileNotFoundError, KeyError):
            logging.exception(cls.__name__)
            raise StructureError(
                "Erreur dans le chargement de la sauvegarde locale !")
        else:
            return cls(cls.decode_json_str(s))