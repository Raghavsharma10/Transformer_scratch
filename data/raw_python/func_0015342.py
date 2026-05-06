def dev_assistant_start(self):
        """
        Thread executes devassistant API.
        """
        #logger_gui.info("Thread run")
        path = self.top_assistant.get_selected_subassistant_path(**self.kwargs)
        kwargs_decoded = dict()
        for k, v in self.kwargs.items():
            kwargs_decoded[k] = \
                v.decode(utils.defenc) if not six.PY3 and isinstance(v, str) else v
        self.dev_assistant_runner = path_runner.PathRunner(path, kwargs_decoded)
        try:
            self.dev_assistant_runner.run()
            Gdk.threads_enter()
            if not self.project_canceled:
                message = '<span color="#008000">Done</span>'
                link = True
                back = False
            else:
                message = '<span color="#FF0000">Failed</span>'
                link = False
                back = True
            self.allow_buttons(message=message, link=link, back=back)
            Gdk.threads_leave()
        except exceptions.ClException as cle:
            msg = replace_markup_chars(cle.message)
            if not six.PY3:
                msg = msg.encode(utils.defenc)
            self.allow_buttons(back=True, link=False,
                               message='<span color="#FF0000">Failed: {0}</span>'.
                               format(msg))
        except exceptions.ExecutionException as exe:
            msg = replace_markup_chars(six.text_type(exe))
            if not six.PY3:
                msg = msg.encode(utils.defenc)
            self.allow_buttons(back=True, link=False,
                               message='<span color="#FF0000">Failed: {0}</span>'.
                               format((msg[:80] + '...') if len(msg) > 80 else msg))
        except IOError as ioe:
            self.allow_buttons(back=True, link=False,
                               message='<span color="#FF0000">Failed: {0}</span>'.
                               format((ioe.message[:80] + '...') if len(ioe.message) > 80 else ioe.message))