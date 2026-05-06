def set_shares(self):
        """
        Setta la variabile membro 'self.samba_shares' il quale e' una lista
        di dizionari con i dati da passare ai comandi di "umount" e "mount".
        I vari dizionari sono popolati o da un file ~/.pygmount.rc e da un
        file passato dall'utente.
        """
        if self.filename is None:
            self.filenamename = os.path.expanduser(
                '~%s/%s' % (self.host_username, FILE_RC))
        if not os.path.exists(self.filename):
            error_msg = (u"Impossibile trovare il file di configurazione "
                         u"'%s'.\nLe unità di rete non saranno collegate." % (
                             FILE_RC.lstrip('.')))
            if not self.shell_mode:
                ErrorMessage(error_msg)
            logging.error(error_msg)
            sys.exit(5)
        if self.verbose:
            logging.warning("File RC utilizzato: %s", self.filename)
        self.samba_shares = read_config(self.filename)