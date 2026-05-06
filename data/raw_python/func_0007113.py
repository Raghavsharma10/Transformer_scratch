def save_to_local(self, callback_etat=print):
        """
        Saved current in memory base to local file.
        It's a backup, not a convenient way to update datas

        :param callback_etat: state callback, taking  str,int,int as args
        """
        callback_etat("Aquisition...", 0, 3)
        d = self.dumps()
        s = json.dumps(d, indent=4, cls=formats.JsonEncoder)
        callback_etat("Chiffrement...", 1, 3)
        s = security.protege_data(s, True)
        callback_etat("Enregistrement...", 2, 3)
        try:
            with open(self.LOCAL_DB_PATH, 'wb') as f:
                f.write(s)
        except (FileNotFoundError):
            logging.exception(self.__class__.__name__)
            raise StructureError("Chemin de sauvegarde introuvable !")