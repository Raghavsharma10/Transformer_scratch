def eval(self, command):
        'Blocking call, returns the value of the execution in JS'
        event = threading.Event()
        # TODO: Add event to server
        #job_id = str(id(command))
        import random
        job_id = str(random.random())
        server.EVALUATIONS[job_id] = event

        message = '?' + job_id + '=' + command
        logging.info(('message:', [message]))
        for listener in server.LISTENERS.get(self.path, []):
            logging.debug(('listener:', listener))
            listener.write_message(message)

        success = event.wait(timeout=30)

        if success:
            value_parser = server.RESULTS[job_id]
            del server.EVALUATIONS[job_id]
            del server.RESULTS[job_id]
            return value_parser()
        else:
            del server.EVALUATIONS[job_id]
            if job_id in server.RESULTS:
                del server.RESULTS[job_id]
            raise IOError('Evaluation failed.')