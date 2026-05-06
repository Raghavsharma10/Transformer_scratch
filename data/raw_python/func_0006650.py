def set_callbacks(self, **dic_functions):
        """Register callbacks needed by the interface object"""
        for action in self.interface.CALLBACKS:
            try:
                f = dic_functions[action]
            except KeyError:
                pass
            else:
                setattr(self.interface.callbacks, action, f)
        manquantes = [
            a for a in self.interface.CALLBACKS if not a in dic_functions]
        if not manquantes:
            logging.debug(
                f"{self.__class__.__name__} : Tous les callbacks demandés sont fournis.")
        else:
            logging.warning(
                f"{self.__class__.__name__} didn't set asked callbacks {manquantes}")