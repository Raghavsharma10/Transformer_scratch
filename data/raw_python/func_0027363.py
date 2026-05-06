def register_handle(self, handle, location, checksum=None, additional_URLs=None, overwrite=False, **extratypes):
        '''
        Registers a new Handle with given name. If the handle already exists
        and overwrite is not set to True, the method will throw an
        exception.

        :param handle: The full name of the handle to be registered (prefix
            and suffix)
        :param location: The URL of the data entity to be referenced
        :param checksum: Optional. The checksum string.
        :param extratypes: Optional. Additional key value pairs.
        :param additional_URLs: Optional. A list of URLs (as strings) to be
            added to the handle record as 10320/LOC entry.
        :param overwrite: Optional. If set to True, an existing handle record
            will be overwritten. Defaults to False.
        :raises: :exc:`~b2handle.handleexceptions.HandleAlreadyExistsException` Only if overwrite is not set or
            set to False.
        :raises: :exc:`~b2handle.handleexceptions.HandleAuthenticationError`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :return: The handle name.
        '''
        LOGGER.debug('register_handle...')

        # If already exists and can't be overwritten:
        if overwrite == False:
            handlerecord_json = self.retrieve_handle_record_json(handle)
            if handlerecord_json is not None:
                msg = 'Could not register handle'
                LOGGER.error(msg + ', as it already exists.')
                raise HandleAlreadyExistsException(handle=handle, msg=msg)

        # Create admin entry
        list_of_entries = []
        adminentry = self.__create_admin_entry(
            self.__handleowner,
            self.__HS_ADMIN_permissions,
            self.__make_another_index(list_of_entries, hs_admin=True),
            handle
        )
        list_of_entries.append(adminentry)

        # Create other entries
        entry_URL = self.__create_entry(
            'URL',
            location,
            self.__make_another_index(list_of_entries, url=True)
        )
        list_of_entries.append(entry_URL)
        if checksum is not None:
            entryChecksum = self.__create_entry(
                'CHECKSUM',
                checksum,
                self.__make_another_index(list_of_entries)
            )
            list_of_entries.append(entryChecksum)
        if extratypes is not None:
            for key, value in extratypes.items():
                entry = self.__create_entry(
                    key,
                    value,
                    self.__make_another_index(list_of_entries)
                )
                list_of_entries.append(entry)
        if additional_URLs is not None and len(additional_URLs) > 0:
            for url in additional_URLs:
                self.__add_URL_to_10320LOC(url, list_of_entries, handle)

        # Create record itself and put to server
        op = 'registering handle'
        resp, put_payload = self.__send_handle_put_request(
            handle,
            list_of_entries,
            overwrite=overwrite,
            op=op
        )
        resp_content = decoded_response(resp)
        if hsresponses.was_handle_created(resp) or hsresponses.handle_success(resp):
            LOGGER.info("Handle registered: " + handle)
            return json.loads(resp_content)['handle']
        elif hsresponses.is_temporary_redirect(resp):
            oldurl = resp.url
            newurl = resp.headers['location']
            raise GenericHandleError(
                operation=op,
                handle=handle,
                response=resp,
                payload=put_payload,
                msg='Temporary redirect from ' + oldurl + ' to ' + newurl + '.'
            )
        elif hsresponses.handle_not_found(resp):
            raise GenericHandleError(
                operation=op,
                handle=handle,
                response=resp,
                payload=put_payload,
                msg='Could not create handle. Possibly you used HTTP instead of HTTPS?'
            )
        else:
            raise GenericHandleError(
                operation=op,
                handle=handle,
                reponse=resp,
                payload=put_payload
            )