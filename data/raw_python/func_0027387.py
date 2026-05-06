def check_if_username_exists(self, username):
        '''
        Check if the username handles exists.

        :param username: The username, in the form index:prefix/suffix
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        :raises: :exc:`~b2handle.handleexceptions.GenericHandleError`
        :return: True. If it does not exist, an exception is raised.

        *Note:* Only the existence of the handle is verified. The existence or
        validity of the index is not checked, because entries containing
        a key are hidden anyway.
        '''
        LOGGER.debug('check_if_username_exists...')

        _, handle = b2handle.utilhandle.remove_index_from_handle(username)

        resp = self.send_handle_get_request(handle)
        resp_content = decoded_response(resp)
        if b2handle.hsresponses.does_handle_exist(resp):
            handlerecord_json = json.loads(resp_content)
            if not handlerecord_json['handle'] == handle:
                raise GenericHandleError(
                    operation='Checking if username exists',
                    handle=handle,
                    reponse=resp,
                    msg='The check returned a different handle than was asked for.'
                )
            return True
        elif b2handle.hsresponses.handle_not_found(resp):
            msg = 'The username handle does not exist'
            raise HandleNotFoundException(handle=handle, msg=msg, response=resp)
        else:
            op = 'checking if handle exists'
            msg = 'Checking if username exists went wrong'
            raise GenericHandleError(operation=op, handle=handle, response=resp, msg=msg)