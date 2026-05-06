def _send_socket_request(self, xml_request):
        """ Send a request via protobuf.

            Args:
                xml_request -- A fully formed xml request string for the CPS.

            Returns:
                The raw xml response string.
        """
        def to_variant(number):
            buff = []
            while number:
                byte = number % 128
                number = number // 128
                if number > 0:
                    byte |= 0x80
                buff.append(chr(byte))
            return ''.join(buff)

        def from_variant(stream):
            used = 0
            number = 0
            q = 1
            while True:
                byte = ord(stream[used])
                used += 1
                number += q * (byte & 0x7F)
                q *= 128
                if byte&0x80==0:
                    break
            return (number, used)

        def encode_fields(fields):
            chunks = []
            for field_id, message in fields.items():
                chunks.append(to_variant((field_id << 3) | 2)) # Hardcoded WireType=2
                chunks.append(to_variant(len(message)))
                chunks.append(message)
            return ''.join(chunks)

        def decode_fields(stream):
            fields = {}
            offset = 0
            stream_lenght = len(stream)
            while offset<stream_lenght:
                field_header, used = from_variant(stream[offset:])
                offset += used
                wire_type = field_header & 0x07
                field_id = field_header >> 3
                if wire_type==2:
                    message_lenght, used = from_variant(stream[offset:])
                    offset += used
                    fields[field_id] = stream[offset:offset+message_lenght]
                    offset += message_lenght
                elif wire_type==0:
                    fields[field_id], used = from_variant(stream[offset:])
                    offset += used
                elif wire_type==1:
                    fields[field_id] = stream[offset:offset+8]
                    offset += 8
                elif wire_type==3:
                    raise ConnectionError()
                elif wire_type==4:
                    raise ConnectionError()
                elif wire_type==5:
                    fields[field_id] = stream[offse:offset+4]
                    offset += 4
                else:
                    raise ConnectionError()
            return fields


        def make_header(lenght):
            result = []
            result.append(chr((lenght & 0x000000FF)))
            result.append(chr((lenght & 0x0000FF00) >> 8))
            result.append(chr((lenght & 0x00FF0000) >> 16))
            result.append(chr((lenght & 0xFF000000) >> 24))
            return '\t\t\x00\x00' + ''.join(result)

        def parse_header(header):
            if len(header) == 8 and header[0] == '\t' and header[1] == '\t' and\
                    header[2] == '\00' and header[3] == '\00':
                return ord(header[4]) | (ord(header[5]) << 8) |\
                        (ord(header[6]) << 16) | (ord(header[7]) << 24)
            else:
                raise ConnectionError()

        def socket_send(data):
            sent_bytes = 0
            failures = 0
            total_bytes = len(data)
            while sent_bytes < total_bytes:
                sent = self._connection.send(data[sent_bytes:])
                if sent == 0:
                    failures += 1
                    if failures > 5:
                        raise ConnectionError()
                    continue
                sent_bytes += sent

        def socket_recieve(lenght):
            total_recieved = 0
            failures = 5
            recieved_chunks = []
            while total_recieved<lenght:
                chunk = self._connection.recv(lenght-total_recieved)
                if not chunk:
                    failures += 1
                    if failures > 5:
                        raise ConnectionError()
                    continue
                recieved_chunks.append(chunk)
                total_recieved += len(chunk)
            return ''.join(recieved_chunks)

        encoded_message = encode_fields({1: xml_request,
                                         2: self._storage if self._storage else "special:detect-storage"})
        header = make_header(len(encoded_message))

        try: # Retry once if failed in case the socket has just gone bad.
            socket_send(header+encoded_message)
        except (ConnectionError, socket.error):
            self._connection.close()
            self._open_connection()
            socket_send(header+encoded_message)

        # TODO: timeout
        header = socket_recieve(8)
        lenght = parse_header(header)
        encoded_response = socket_recieve(lenght)
        response = decode_fields(encoded_response)
        # TODO: Test for id=3 error message
        # TODO: check for and raise errors
        return response[1]