def request_generic(self, act, coro, perform, complete):
        """
        Performs an overlapped request (via `perform` callable) and saves
        the token and the (`overlapped`, `perform`, `complete`) trio.
        """
        overlapped = OVERLAPPED()
        overlapped.object = act
        self.add_token(act, coro, (overlapped, perform, complete))

        rc, nbytes = perform(act, overlapped)
        completion_key = c_long(0)
        if rc == 0:
            # ah geez, it didn't got in the iocp, we have a result!
            pass


            # ok this is weird, apparently this doesn't need to be requeued
            #  - need to investigate why (TODO)
            #~ PostQueuedCompletionStatus(
                #~ self.iocp, # HANDLE CompletionPort
                #~ nbytes, # DWORD dwNumberOfBytesTransferred
                #~ byref(completion_key), # ULONG_PTR dwCompletionKey
                #~ overlapped # LPOVERLAPPED lpOverlapped
            #~ )
        elif rc != WSA_IO_PENDING:
            self.remove_token(act)
            raise SocketError(rc, "%s on %r" % (ctypes.FormatError(rc), act))