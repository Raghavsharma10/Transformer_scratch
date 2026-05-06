def add_additional_URL(self, handle, *urls, **attributes):
        '''
        Add a URL entry to the handle record's 10320/LOC entry. If 10320/LOC
        does not exist yet, it is created. If the 10320/LOC entry already
        contains the URL, it is not added a second time.

        :param handle: The handle to add the URL to.
        :param urls: The URL(s) to be added. Several URLs may be specified.
        :param attributes: Optional. Additional key-value pairs to set as
            attributes to the <location> elements, e.g. weight, http_role or
            custom attributes. Note: If the URL already exists but the
            attributes are different, they are updated!
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :raises: :exc:`~b2handle.handleexceptions.HandleAuthenticationError`
        '''
        LOGGER.debug('add_additional_URL...')

        handlerecord_json = self.retrieve_handle_record_json(handle)
        if handlerecord_json is None:
            msg = 'Cannot add URLS to unexisting handle!'
            raise HandleNotFoundException(handle=handle, msg=msg)
        list_of_entries = handlerecord_json['values']

        is_new = False
        for url in urls:
            if not self.is_URL_contained_in_10320LOC(handle, url, handlerecord_json):
                is_new = True

        if not is_new:
            LOGGER.debug("add_additional_URL: No new URL to be added (so no URL is added at all).")
        else:

            for url in urls:
                self.__add_URL_to_10320LOC(url, list_of_entries, handle)

            op = 'adding URLs'
            resp, put_payload = self.__send_handle_put_request(handle, list_of_entries, overwrite=True, op=op)
            # TODO FIXME (one day) Overwrite by index.

            if hsresponses.handle_success(resp):
                pass
            else:
                msg = 'Could not add URLs ' + str(urls)
                raise GenericHandleError(
                    operation=op,
                    handle=handle,
                    reponse=resp,
                    msg=msg,
                    payload=put_payload
                )