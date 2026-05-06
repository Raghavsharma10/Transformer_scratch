def run(self):
        """ Main event loop """

        # Show stream list
        self.show_streams()

        while True:
            self.s.refresh()

            # See if any stream has ended
            self.check_stopped_streams()

            # Wait on stdin or on the streams output
            souts = self.q.get_stdouts()
            souts.append(sys.stdin)
            try:
                (r, w, x) = select.select(souts, [], [], 1)
            except select.error:
                continue
            if not r:
                if self.config.CHECK_ONLINE_INTERVAL <= 0: continue
                cur_time = int(time())
                time_delta = cur_time - self.last_autocheck
                if time_delta > self.config.CHECK_ONLINE_INTERVAL:
                    self.check_online_streams()
                    self.set_status('Next check at {0}'.format(
                        strftime('%H:%M:%S', localtime(time() + self.config.CHECK_ONLINE_INTERVAL))
                        ))
                continue
            for fd in r:
                if fd != sys.stdin:
                    # Set the new status line only if non-empty
                    msg = fd.readline()
                    if msg:
                        self.set_status(msg[:-1])
                else:
                    # Main event loop
                    c = self.pads[self.current_pad].getch()
                    if c == curses.KEY_UP or c == ord('k'):
                        self.move(-1)
                    elif c == curses.KEY_DOWN or c == ord('j'):
                        self.move(1)
                    elif c == ord('f'):
                        if self.current_pad == 'streams':
                            self.filter_streams()
                    elif c == ord('F'):
                        if self.current_pad == 'streams':
                            self.clear_filter()
                    elif c == ord('g'):
                        if self.got_g:
                            self.move(0, absolute=True)
                            self.got_g = False
                            continue
                        self.got_g = True
                    elif c == ord('G'):
                        self.move(len(self.filtered_streams)-1, absolute=True)
                    elif c == ord('q'):
                        if self.current_pad == 'streams':
                            self.q.terminate()
                            return
                        else:
                            self.show_streams()
                    elif c == 27: # ESC
                        if self.current_pad != 'streams':
                            self.show_streams()
                    if self.current_pad == 'help':
                        continue
                    elif c == 10:
                        self.play_stream()
                    elif c == ord('s'):
                        self.stop_stream()
                    elif c == ord('c'):
                        self.reset_stream()
                    elif c == ord('n'):
                        self.edit_stream('name')
                    elif c == ord('r'):
                        self.edit_stream('res')
                    elif c == ord('u'):
                        self.edit_stream('url')
                    elif c == ord('l'):
                        self.show_commandline()
                    elif c == ord('L'):
                        self.shift_commandline()
                    elif c == ord('a'):
                        self.prompt_new_stream()
                    elif c == ord('d'):
                        self.delete_stream()
                    elif c == ord('o'):
                        self.show_offline_streams ^= True
                        self.refilter_streams()
                    elif c == ord('O'):
                        self.check_online_streams()
                    elif c == ord('h') or c == ord('?'):
                        self.show_help()