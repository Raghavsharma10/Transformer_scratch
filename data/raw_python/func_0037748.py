def create_crud_func(self, method, request_type="CRUD"):
        """
        create_crud_func
        """
        def _crud(resource,
                  data=None,
                  block=True,
                  timeout=60,
                  topic="/controller",
                  tunnel=None,
                  qos=0):
            """
            _crud

            block
                True: wait until response arrival
                False: wait until message is already published to local broker
            """
            headers = {
                "resource": resource,
                "method": method
            }

            # DIRECT message needs put tunnel in headers for controller
            if request_type == "DIRECT":
                if tunnel is not None:
                    headers["tunnel"] = tunnel
                elif self._conn.tunnels["view"][0] is not None:
                    headers["tunnel"] = self._conn.tunnels["view"][0]
                elif self._conn.tunnels["model"][0] is not None:
                    headers["tunnel"] = self._conn.tunnels["model"][0]
                else:
                    headers["tunnel"] = self._conn.tunnels["internel"][0]

            message = self._create_message(headers, data)
            with self._session.session_lock:
                mid = self._conn.publish(topic=topic,
                                         qos=qos,
                                         payload=message.to_dict())
                session = self._session.create(message, mid=mid, age=timeout)
                session["status"] = Status.SENDING

            # blocking, until we get response or published
            if block is False:
                return self._wait_published(session)
            return self._wait_resolved(session)
        return _crud