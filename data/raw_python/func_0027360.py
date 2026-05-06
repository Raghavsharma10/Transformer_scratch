def exchange_additional_URL(self, handle, old, new):
        '''
        Exchange an URL in the 10320/LOC entry against another, keeping the same id
        and other attributes.

        :param handle: The handle to modify.
        :param old: The URL to replace.
        :param new: The URL to set as new URL.
        '''
        LOGGER.debug('exchange_additional_URL...')

        handlerecord_json = self.retrieve_handle_record_json(handle)
        if handlerecord_json is None:
            msg = 'Cannot exchange URLs in unexisting handle'
            raise HandleNotFoundException(
                handle=handle,
                msg=msg
            )
        list_of_entries = handlerecord_json['values']

        if not self.is_URL_contained_in_10320LOC(handle, old, handlerecord_json):
            LOGGER.debug('exchange_additional_URL: No URLs exchanged, as the url was not in the record.')
        else:
            self.__exchange_URL_in_13020loc(old, new, list_of_entries, handle)

            op = 'exchanging URLs'
            resp, put_payload = self.__send_handle_put_request(
                handle,
                list_of_entries,
                overwrite=True,
                op=op
            )
            # TODO FIXME (one day): Implement overwriting by index (less risky)
            if hsresponses.handle_success(resp):
                pass
            else:
                msg = 'Could not exchange URL ' + str(old) + ' against ' + str(new)
                raise GenericHandleError(
                    operation=op,
                    handle=handle,
                    reponse=resp,
                    msg=msg,
                    payload=put_payload
                )