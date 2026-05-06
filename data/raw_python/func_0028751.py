def run(self):
        """
        Process the input queues in lock-step, and push any results to
        the registered output queues.

        """
        try:
            while True:
                input_chunks = [input.get() for input in self.input_queues]
                for input in self.input_queues:
                    input.task_done()
                if any(chunk is QUEUE_ABORT for chunk in input_chunks):
                    self.abort()
                    return
                if any(chunk is QUEUE_FINISHED for chunk in input_chunks):
                    break
                self.output(self.process_chunks(input_chunks))
            # Finalise the final chunk (process_chunks does this for all
            # but the last chunk).
            self.output(self.finalise())
        except:
            self.abort()
            raise
        else:
            for queue in self.output_queues:
                queue.put(QUEUE_FINISHED)