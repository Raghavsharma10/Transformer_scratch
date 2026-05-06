def post_and_track(self, url, payload, refresh_rate_sec=1):
        """
        Post and track progress, displaying progress bars.

        May display the wrong progress if 2 things post/put on the same
        procedure name at the same time.
        """
        if not url.startswith('/v1/procedures'):
            raise Exception("The only supported route is /v1/procedures")
        if url.endswith('/runs'):
            raise Exception(
                "Posting and tracking run is unsupported at the moment")
        if len(url.split('/')) != 3:
            raise Exception("You must POST a procedure")

        if 'params' not in payload:
            payload['params'] = {}
        payload['params']['runOnCreation'] = False

        res = self.post('/v1/procedures', payload).json()
        proc_id = res['id']

        pm = ProgressMonitor(self, refresh_rate_sec, proc_id,
                             notebook=self.notebook)

        t = threading.Thread(target=pm.monitor_progress)
        t.start()

        try:
            return self.post('/v1/procedures/{}/runs'.format(proc_id), {})
        except Exception as e:
            print(e)
        finally:
            pm.event.set()
            t.join()