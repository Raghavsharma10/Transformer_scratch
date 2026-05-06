def broadcast(self, clients, msg):
        """ Optimized C{broadcast} implementation. Depending on type of the
        session, will json-encode message once and will call either
        C{send_message} or C{send_jsonifed}.

        @param clients: Clients iterable

        @param msg: Message to send
        """
        json_msg = None

        count = 0

        for c in clients:
            sess = c.session
            if not sess.is_closed:
                if sess.send_expects_json:
                    if json_msg is None:
                        json_msg = proto.json_encode(msg)
                    sess.send_jsonified(json_msg, stats=False)
                else:
                    sess.send_message(msg, stats=False)

                count += 1

        self.stats.packSent(count)