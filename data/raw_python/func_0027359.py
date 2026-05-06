def delete_handle(self, handle, *other):
        '''Delete the handle and its handle record. If the Handle is not found, an Exception is raised.

        :param handle: Handle to be deleted.
        :param other: Deprecated. This only exists to catch wrong method usage
            by users who are used to delete handle VALUES with the method.
        :raises: :exc:`~b2handle.handleexceptions.HandleAuthenticationError`
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        '''

        LOGGER.debug('delete_handle...')

        utilhandle.check_handle_syntax(handle)

        # Safety check. In old epic client, the method could be used for
        # deleting handle values (not entire handle) by specifying more
        # parameters.
        if len(other) > 0:
            message = 'You specified more than one argument. If you wanted' + \
                ' to delete just some values from a handle, please use the' + \
                ' new method "delete_handle_value()".'
            raise TypeError(message)

        op = 'deleting handle'
        resp = self.__send_handle_delete_request(handle, op=op)
        if hsresponses.handle_success(resp):
            LOGGER.info('Handle ' + handle + ' deleted.')
        elif hsresponses.handle_not_found(resp):
            msg = ('delete_handle: Handle ' + handle + ' did not exist, '
                   'so it could not be deleted.')
            LOGGER.debug(msg)
            raise HandleNotFoundException(msg=msg, handle=handle, response=resp)
        else:
            raise GenericHandleError(op=op, handle=handle, response=resp)