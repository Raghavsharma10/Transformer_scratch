def wait_for_shutdown_signal(
        self,
        please_stop=False,  # ASSIGN SIGNAL TO STOP EARLY
        allow_exit=False,  # ALLOW "exit" COMMAND ON CONSOLE TO ALSO STOP THE APP
        wait_forever=True  # IGNORE CHILD THREADS, NEVER EXIT.  False => IF NO CHILD THREADS LEFT, THEN EXIT
    ):
        """
        FOR USE BY PROCESSES THAT NEVER DIE UNLESS EXTERNAL SHUTDOWN IS REQUESTED

        CALLING THREAD WILL SLEEP UNTIL keyboard interrupt, OR please_stop, OR "exit"

        :param please_stop:
        :param allow_exit:
        :param wait_forever:: Assume all needed threads have been launched. When done
        :return:
        """
        self_thread = Thread.current()
        if self_thread != MAIN_THREAD or self_thread != self:
            Log.error("Only the main thread can sleep forever (waiting for KeyboardInterrupt)")

        if isinstance(please_stop, Signal):
            # MUTUAL SIGNALING MAKES THESE TWO EFFECTIVELY THE SAME SIGNAL
            self.please_stop.on_go(please_stop.go)
            please_stop.on_go(self.please_stop.go)
        else:
            please_stop = self.please_stop

        if not wait_forever:
            # TRIGGER SIGNAL WHEN ALL CHILDREN THEADS ARE DONE
            with self_thread.child_lock:
                pending = copy(self_thread.children)
            children_done = AndSignals(please_stop, len(pending))
            children_done.signal.on_go(self.please_stop.go)
            for p in pending:
                p.stopped.on_go(children_done.done)

        try:
            if allow_exit:
                _wait_for_exit(please_stop)
            else:
                _wait_for_interrupt(please_stop)
        except KeyboardInterrupt as _:
            Log.alert("SIGINT Detected!  Stopping...")
        except SystemExit as _:
            Log.alert("SIGTERM Detected!  Stopping...")
        finally:
            self.stop()