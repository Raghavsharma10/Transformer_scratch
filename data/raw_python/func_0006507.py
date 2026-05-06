def load_remote_data(self, callback_etat=print):
        """
        Load remote data. On succes, build base.
        On failure, raise :class:`~.Core.exceptions.StructureError`, :class:`~.Core.exceptions.ConnexionError`

        :param callback_etat: State renderer str , int , int -> None
        """
        callback_etat("Chargement des utilisateurs", 0, 1)
        self._load_users()
        self.base = self.BASE_CLASS.load_from_db(callback_etat=callback_etat)