def requirements(self):
        """
        Verifica che tutti i pacchetti apt necessari al "funzionamento" della
        classe siano installati. Se cosi' non fosse li installa.
        """
        cache = apt.cache.Cache()
        for pkg in self.pkgs_required:
            try:
                pkg = cache[pkg]
                if not pkg.is_installed:
                    try:
                        pkg.mark_install()
                        cache.commit()
                    except LockFailedException as lfe:
                        logging.error(
                            'Errore "{}" probabilmente l\'utente {} non ha i '
                            'diritti di amministratore'.format(lfe,
                                                               self.username))
                        raise lfe
                    except Exception as e:
                        logging.error('Errore non classificato "{}"'.format(e))
                        raise e
            except KeyError:
                logging.error('Il pacchetto "{}" non e\' presente in questa'
                              ' distribuzione'.format(pkg))