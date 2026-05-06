def put_and_track(self, url, payload, refresh_rate_sec=1):
        """
        Put and track progress, displaying progress bars.

        May display the wrong progress if 2 things post/put on the same
        procedure name at the same time.
        """
        if not url.startswith('/v1/procedures'):
            raise Exception("The only supported route is /v1/procedures")
        parts = url.split('/')
        len_parts = len(parts)
        if len_parts not in [4, 6]:
            raise Exception(
                "You must either PUT a procedure or a procedure run")

        proc_id = parts[3]
        run_id = None

        if len_parts == 4:
                if 'params' not in payload:
                    payload['params'] = {}
                payload['params']['runOnCreation'] = True
        elif len_parts == 6:
            run_id = parts[-1]

        pm = ProgressMonitor(self, refresh_rate_sec, proc_id, run_id,
                             self.notebook)
        t = threading.Thread(target=pm.monitor_progress)
        t.start()

        try:
            return self.put(url, payload)
        except Exception as e:
            print(e)
        finally:
            pass
            pm.event.set()
            t.join()