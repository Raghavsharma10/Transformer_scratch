def onSetup(self, request_type, request, value, index, length):
        """
        Called when a setup USB transaction was received.

        Default implementation:
        - handles USB_REQ_GET_STATUS on interface and endpoints
        - handles USB_REQ_CLEAR_FEATURE(USB_ENDPOINT_HALT) on endpoints
        - handles USB_REQ_SET_FEATURE(USB_ENDPOINT_HALT) on endpoints
        - halts on everything else

        If this method raises anything, endpoint 0 is halted by its caller and
        exception is let through.

        May be overridden in subclass.
        """
        if (request_type & ch9.USB_TYPE_MASK) == ch9.USB_TYPE_STANDARD:
            recipient = request_type & ch9.USB_RECIP_MASK
            is_in = (request_type & ch9.USB_DIR_IN) == ch9.USB_DIR_IN
            if request == ch9.USB_REQ_GET_STATUS:
                if is_in and length == 2:
                    if recipient == ch9.USB_RECIP_INTERFACE:
                        if value == 0:
                            status = 0
                            if index == 0:
                                if self.function_remote_wakeup_capable:
                                    status |= 1 << 0
                                if self.function_remote_wakeup:
                                    status |= 1 << 1
                            self.ep0.write(struct.pack('<H', status)[:length])
                            return
                    elif recipient == ch9.USB_RECIP_ENDPOINT:
                        if value == 0:
                            try:
                                endpoint = self.getEndpoint(index)
                            except IndexError:
                                pass
                            else:
                                status = 0
                                if endpoint.isHalted():
                                    status |= 1 << 0
                                self.ep0.write(
                                    struct.pack('<H', status)[:length],
                                )
                                return
            elif request == ch9.USB_REQ_CLEAR_FEATURE:
                if not is_in and length == 0:
                    if recipient == ch9.USB_RECIP_ENDPOINT:
                        if value == ch9.USB_ENDPOINT_HALT:
                            try:
                                endpoint = self.getEndpoint(index)
                            except IndexError:
                                pass
                            else:
                                endpoint.clearHalt()
                                self.ep0.read(0)
                                return
                    elif recipient == ch9.USB_RECIP_INTERFACE:
                        if value == ch9.USB_INTRF_FUNC_SUSPEND:
                            if self.function_remote_wakeup_capable:
                                self.disableRemoteWakeup()
                                self.ep0.read(0)
                                return
            elif request == ch9.USB_REQ_SET_FEATURE:
                if not is_in and length == 0:
                    if recipient == ch9.USB_RECIP_ENDPOINT:
                        if value == ch9.USB_ENDPOINT_HALT:
                            try:
                                endpoint = self.getEndpoint(index)
                            except IndexError:
                                pass
                            else:
                                endpoint.halt()
                                self.ep0.read(0)
                                return
                    elif recipient == ch9.USB_RECIP_INTERFACE:
                        if value == ch9.USB_INTRF_FUNC_SUSPEND:
                            if self.function_remote_wakeup_capable:
                                self.enableRemoteWakeup()
                                self.ep0.read(0)
                                return
        self.ep0.halt(request_type)