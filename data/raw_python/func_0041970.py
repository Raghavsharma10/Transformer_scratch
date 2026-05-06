def run(self):
        """
        Esegue il montaggio delle varie condivisioni chiedendo all'utente
        username e password di dominio.
        """
        logging.info('start run with "{}" at {}'.format(
            self.username, datetime.datetime.now()))
        progress = Progress(text="Controllo requisiti software...",
                            pulsate=True, auto_close=True)
        progress(1)
        try:
            self.requirements()
        except LockFailedException as lfe:
            ErrorMessage('Errore "{}" probabilmente l\'utente {} non ha i'
                         ' diritti di amministratore'.format(lfe,
                                                             self.username))
            sys.exit(20)
        except Exception as e:
            ErrorMessage("Si e' verificato un errore generico: {}".format(e))
            sys.exit(21)
        progress(100)

        self.set_shares()
        # richiesta username del dominio
        insert_msg = "Inserisci l'utente del Dominio/Posta Elettronica"
        default_username = (self.host_username if self.host_username
                            else os.environ['USER'])
        self.domain_username = GetText(text=insert_msg,
                                       entry_text=self.username)

        if self.domain_username is None or len(self.domain_username) == 0:
            error_msg = "Inserimento di un username di dominio vuoto"
            ErrorMessage(self.msg_error % error_msg)
            sys.exit(2)

        # richiesta della password di dominio
        insert_msg = u"Inserisci la password del Dominio/Posta Elettronica"
        self.domain_password = GetText(text=insert_msg,
                                       entry_text='password',
                                       password=True)

        if self.domain_password is None or len(self.domain_password) == 0:
            error_msg = u"Inserimento di una password di dominio vuota"
            ErrorMessage(self.msg_error % error_msg)
            sys.exit(3)

        progress_msg = u"Collegamento unità di rete in corso..."
        progress = Progress(text=progress_msg,
                            pulsate=True,
                            auto_close=True)
        progress(1)
        # ciclo per montare tutte le condivisioni
        result = []
        for share in self.samba_shares:
            # print("#######")
            # print(share)
            if 'mountpoint' not in share.keys():
                # creazione stringa che rappresente il mount-point locale
                mountpoint = os.path.expanduser(
                    '~%s/%s/%s' % (self.host_username,
                                   share['hostname'],
                                   share['share']))
                share.update({'mountpoint': mountpoint})
            elif not share['mountpoint'].startswith('/'):
                mountpoint = os.path.expanduser(
                    '~%s/%s' % (self.host_username, share['mountpoint']))
                share.update({'mountpoint': mountpoint})

            share.update({
                'host_username': self.host_username,
                'domain_username': share.get(
                    'username', self.domain_username),
                'domain_password': share.get(
                    'password', self.domain_password)})

            # controllo che il mount-point locale esista altrimenti non
            # viene creato
            if not os.path.exists(share['mountpoint']):
                if self.verbose:
                    logging.warning('Mountpoint "%s" not exist.' %
                                    share['mountpoint'])
                if not self.dry_run:
                    os.makedirs(share['mountpoint'])

            # smonto la condivisione prima di rimontarla
            umont_cmd = self.cmd_umount % share
            if self.verbose:
                logging.warning("Umount command: %s" % umont_cmd)
            if not self.dry_run:
                umount_p = subprocess.Popen(umont_cmd,
                                            shell=True)
                returncode = umount_p.wait()
                time.sleep(2)

            mount_cmd = self.cmd_mount % share
            if self.verbose:
                placeholder = ",password="
                logging.warning("Mount command: %s%s" % (mount_cmd.split(
                    placeholder)[0], placeholder + "******\""))

            # print(mount_cmd)
            # print("#######")
            if not self.dry_run:
                # montaggio della condivisione
                p_mnt = subprocess.Popen(mount_cmd, shell=True,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                returncode = p_mnt.wait()
                result.append({'share': share['share'],
                               'returncode': returncode,
                               'stdout': p_mnt.stdout.read(),
                               'stderr': p_mnt.stderr.read()})
        progress(100)
        if self.verbose:
            logging.warning("Risultati: %s" % result)