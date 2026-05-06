def update_slidepos(self):
        """
        Periodically update the slide position.

        Also farmed out to a thread to avoid hanging GUI main thread
        """
        g = get_root(self).globals
        if not g.cpars['focal_plane_slide_on']:
            self.after(20000, self.update_slidepos)
            return

        def slide_threaded_update():
            try:
                (pos_ms, pos_mm, pos_px), msg = g.fpslide.slide.return_position()
                self.slide_pos_queue.put((pos_ms, pos_mm, pos_px))
            except Exception as err:
                t, v, tb = sys.exc_info()
                error = traceback.format_exception_only(t, v)[0].strip()
                tback = 'Slide Traceback (most recent call last):\n' + \
                        ''.join(traceback.format_tb(tb))
                g.FIFO.put(('Slide', error, tback))

        t = threading.Thread(target=slide_threaded_update)
        t.start()
        self.after(20000, self.update_slidepos)