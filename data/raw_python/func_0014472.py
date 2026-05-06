def _send_kex_init(self):
        """
        announce to the other side that we'd like to negotiate keys, and what
        kind of key negotiation we support.
        """
        self.clear_to_send_lock.acquire()
        try:
            self.clear_to_send.clear()
        finally:
            self.clear_to_send_lock.release()
        self.in_kex = True
        if self.server_mode:
            if (self._modulus_pack is None) and ('diffie-hellman-group-exchange-sha1' in self._preferred_kex):
                # can't do group-exchange if we don't have a pack of potential primes
                pkex = list(self.get_security_options().kex)
                pkex.remove('diffie-hellman-group-exchange-sha1')
                self.get_security_options().kex = pkex
            available_server_keys = filter(self.server_key_dict.keys().__contains__,
                                           self._preferred_keys)
        else:
            available_server_keys = self._preferred_keys

        m = Message()
        m.add_byte(chr(MSG_KEXINIT))
        m.add_bytes(rng.read(16))
        m.add_list(self._preferred_kex)
        m.add_list(available_server_keys)
        m.add_list(self._preferred_ciphers)
        m.add_list(self._preferred_ciphers)
        m.add_list(self._preferred_macs)
        m.add_list(self._preferred_macs)
        m.add_list(self._preferred_compression)
        m.add_list(self._preferred_compression)
        m.add_string('')
        m.add_string('')
        m.add_boolean(False)
        m.add_int(0)
        # save a copy for later (needed to compute a hash)
        self.local_kex_init = str(m)
        self._send_message(m)