def load_from_db(cls, callback_etat=print, out=None):
        """Launch data fetching then load data received.
        The method _load_remote_db should be overridden.
        If out is given, datas are set in it, instead of returning a new base object.
        """
        dic = cls._load_remote_db(callback_etat)
        callback_etat("Chargement...", 2, 3)
        if out is None:
            return cls(dic)
        cls.__init__(out, datas=dic)