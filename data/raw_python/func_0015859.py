def set_password(self, service, username, password):
        """Set password for the username of the service
        """
        password = self._encrypt(password or '')
        keyring_working_copy = copy.deepcopy(self._keyring)
        service_entries = keyring_working_copy.get(service)
        if not service_entries:
            service_entries = {}
            keyring_working_copy[service] = service_entries
        service_entries[username] = password
        save_result = self._save_keyring(keyring_working_copy)
        if save_result == self.OK:
            self._keyring_dict = keyring_working_copy
            return
        elif save_result == self.CONFLICT:
            # check if we can avoid updating
            self.docs_entry, keyring_dict = self._read()
            existing_pwd = self._get_entry(self._keyring, service, username)
            conflicting_pwd = self._get_entry(keyring_dict, service, username)
            if conflicting_pwd == password:
                # if someone else updated it to the same value then we are done
                self._keyring_dict = keyring_working_copy
                return
            elif conflicting_pwd is None or conflicting_pwd == existing_pwd:
                # if doesn't already exist or is unchanged then update it
                new_service_entries = keyring_dict.get(service, {})
                new_service_entries[username] = password
                keyring_dict[service] = new_service_entries
                save_result = self._save_keyring(keyring_dict)
                if save_result == self.OK:
                    self._keyring_dict = keyring_dict
                    return
                else:
                    raise errors.PasswordSetError(
                        'Failed write after conflict detected')
            else:
                raise errors.PasswordSetError(
                    'Conflict detected, service:%s and username:%s was '
                    'set to a different value by someone else' % (
                        service,
                        username,
                    ),
                )

        raise errors.PasswordSetError('Could not save keyring')