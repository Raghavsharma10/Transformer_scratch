def process_op(self, rc, nbytes, overlap):
        """
        Handles the possible completion or re-queueing if conditions haven't
        been met (the `complete` callable returns false) of a overlapped request.
        """
        act = overlap.object
        overlap.object = None
        if act in self.tokens:
            ol, perform, complete = self.tokens[act]
            assert ol is overlap
            if rc == 0:
                ract = self.try_run_act(act, complete, rc, nbytes)
                if ract:
                    del self.tokens[act]
                    win32file.CancelIo(act.sock._fd.fileno())
                    return ract, act.coro
                else:
                    # operation hasn't completed yet (not enough data etc)
                    # read it in the iocp
                    self.request_generic(act, act.coro, perform, complete)


            else:
                #looks like we have a problem, forward it to the coroutine.

                # this needs some research: ERROR_NETNAME_DELETED, need to reopen
                #the accept sock ?! something like:
                #    warnings.warn("ERROR_NETNAME_DELETED: %r. Re-registering operation." % op)
                #    self.registered_ops[op] = self.run_iocp(op, coro)
                del self.tokens[act]
                win32file.CancelIo(act.sock._fd.fileno())
                return CoroutineException((
                    SocketError, SocketError(
                        (rc, "%s on %r" % (ctypes.FormatError(rc), act))
                    )
                )), act.coro
        else:
            import warnings
            warnings.warn("Unknown token %s" % act)