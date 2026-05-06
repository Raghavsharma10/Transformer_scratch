def _run(self):
        '''Continually poll TWS'''
        stop = self._stop_evt
        connected = self._connected_evt
        tws = self._tws

        fd = tws.fd()
        pollfd = [fd]

        while not stop.is_set():
            while (not connected.is_set() or not tws.isConnected()) and not stop.is_set():
                connected.clear()
                backoff = 0
                retries = 0

                while not connected.is_set() and not stop.is_set():
                    if tws.reconnect_auto and not tws.reconnect():
                        if backoff < self.MAX_BACKOFF:
                            retries += 1
                            backoff = min(2**(retries + 1), self.MAX_BACKOFF)
                        connected.wait(backoff / 1000.)
                    else:
                        connected.wait(1)
                fd = tws.fd()
                pollfd = [fd]

            if fd > 0:
                try:
                    evtin, _evtout, evterr = select.select(pollfd, [], pollfd, 1)
                except select.error:
                    connected.clear()
                    continue
                else:
                    if fd in evtin:
                        try:
                            if not tws.checkMessages():
                                tws.eDisconnect(stop_polling=False)
                                continue
                        except (SystemExit, SystemError, KeyboardInterrupt):
                            break
                        except:
                            try:
                                self._wrapper.pyError(*sys.exc_info())
                            except:
                                print_exc()
                    elif fd in evterr:
                        connected.clear()
                        continue