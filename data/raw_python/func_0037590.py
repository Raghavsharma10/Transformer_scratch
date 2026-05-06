def debug_application(self, environ, start_response):
        """Run the application and conserve the traceback frames."""
        app_iter = None
        try:
            app_iter = self.app(environ, start_response)
            for item in app_iter:
                yield item
            if hasattr(app_iter, 'close'):
                app_iter.close()
        except Exception:
            if hasattr(app_iter, 'close'):
                app_iter.close()

            context = RequestContext({'environ':dict(environ)})
            for injector in self.context_injectors:
                context.update(injector(environ))

            traceback = get_current_traceback(skip=1, show_hidden_frames=self.show_hidden_frames,
                                              context=context)
            for frame in traceback.frames:
                self.frames[frame.id] = frame
            self.tracebacks[traceback.id] = traceback

            try:
                start_response('500 INTERNAL SERVER ERROR', [
                    ('Content-Type', 'text/html; charset=utf-8'),
                    # Disable Chrome's XSS protection, the debug
                    # output can cause false-positives.
                    ('X-XSS-Protection', '0'),
                    ])
            except Exception:
                # if we end up here there has been output but an error
                # occurred.  in that situation we can do nothing fancy any
                # more, better log something into the error log and fall
                # back gracefully.
                environ['wsgi.errors'].write(
                    'Debugging middleware caught exception in streamed '
                    'response at a point where response headers were already '
                    'sent.\n')
            else:
                yield traceback.render_full(
                    evalex=self.evalex,
                    secret=self.secret
                ).encode('utf-8', 'replace')

            # This will lead to double logging in case backlash logger is set to DEBUG
            # but this is actually wanted as some environments, like WebTest, swallow
            # wsgi.environ making the traceback totally disappear.
            log.debug(traceback.plaintext)
            traceback.log(environ['wsgi.errors'])