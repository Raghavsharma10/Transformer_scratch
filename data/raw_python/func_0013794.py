def encrypt_file(self, inpath, force_nocompress=False, force_compress=False, armored=False, checksum=False):
        """public method for single file encryption with optional compression, ASCII armored formatting, and file hash digest generation"""
        if armored:
            if force_compress:
                command_stub = self.command_maxcompress_armored
            elif force_nocompress:
                command_stub = self.command_nocompress_armored
            else:
                if self._is_compress_filetype(inpath):
                    command_stub = self.command_default_armored
                else:
                    command_stub = self.command_nocompress_armored
        else:
            if force_compress:
                command_stub = self.command_maxcompress
            elif force_nocompress:
                command_stub = self.command_nocompress
            else:
                if self._is_compress_filetype(inpath):
                    command_stub = self.command_default
                else:
                    command_stub = self.command_nocompress

        encrypted_outpath = self._create_outfilepath(inpath)
        system_command = command_stub + encrypted_outpath + " --passphrase " + quote(self.passphrase) + " --symmetric " + quote(inpath)

        try:
            response = muterun(system_command)
            # check returned status code
            if response.exitcode == 0:
                stdout(encrypted_outpath + " was generated from " + inpath)
                if checksum:  # add a SHA256 hash digest of the encrypted file - requested by user --hash flag in command
                    from crypto.library import hash
                    encrypted_file_hash = hash.generate_hash(encrypted_outpath)
                    if len(encrypted_file_hash) == 64:
                        stdout("SHA256 hash digest for " + encrypted_outpath + " :")
                        stdout(encrypted_file_hash)
                    else:
                        stdout("Unable to generate a SHA256 hash digest for the file " + encrypted_outpath)
            else:
                stderr(response.stderr, 0)
                stderr("Encryption failed")
                sys.exit(1)
        except Exception as e:
            stderr("There was a problem with the execution of gpg. Encryption failed. Error: [" + str(e) + "]")
            sys.exit(1)