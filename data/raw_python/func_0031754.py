def send_message(self, target_app, dictionary):
        """
        Send a message to the given app, which should be currently running on the Pebble (unless using a non-standard
        AppMessage endpoint, in which case its rules apply).

        AppMessage can only represent flat dictionaries with integer keys; as such, ``dictionary`` must be flat and have
        integer keys.

        Because the AppMessage dictionary type is more expressive than Python's native types allow, all entries in the
        dictionary provided must be wrapped in one of the value types:

        =======================  =============  ============
        AppMessageService type    C type        Python type
        =======================  =============  ============
        :class:`Uint8`           ``uint8_t``    :any:`int`
        :class:`Uint16`          ``uint16_t``   :any:`int`
        :class:`Uint32`          ``uint32_t``   :any:`int`
        :class:`Int8`            ``int8_t``     :any:`int`
        :class:`Int16`           ``int16_t``    :any:`int`
        :class:`Int32`           ``int32_t``    :any:`int`
        :class:`CString`         ``char *``     :any:`str`
        :class:`ByteArray`       ``uint8_t *``  :any:`bytes`
        =======================  =============  ============

        For instance: ::

           appmessage.send_message(UUID("6FEAF2DE-24FA-4ED3-AF66-C853FA6E9C3C"), {
               16: Uint8(62),
               6428356: CString("friendship"),
           })

        :param target_app: The UUID of the app to which to send a message.
        :type target_app: ~uuid.UUID
        :param dictionary: The dictionary to send.
        :type dictionary: dict
        :return: The transaction ID sent message, as used in the ``ack`` and ``nack`` events.
        :rtype: int
        """
        tid = self._get_txid()
        message = self._message_type(transaction_id=tid)
        tuples = []
        for k, v in iteritems(dictionary):
            if isinstance(v, AppMessageNumber):
                tuples.append(AppMessageTuple(key=k, type=v.type,
                                data=struct.pack(self._type_mapping[v.type, v.length], v.value)))
            elif v.type == AppMessageTuple.Type.CString:
                tuples.append(AppMessageTuple(key=k, type=v.type, data=v.value.encode('utf-8') + b'\x00'))
            elif v.type == AppMessageTuple.Type.ByteArray:
                tuples.append(AppMessageTuple(key=k, type=v.type, data=v.value))
        message.data = AppMessagePush(uuid=target_app, dictionary=tuples)
        self._pending_messages[tid] = target_app
        self._pebble.send_packet(message)
        return tid