def reset_everything(self, payload):
        """Kill all processes, delete the queue and clean everything up."""
        kill_signal = signals['9']
        self.process_handler.kill_all(kill_signal, True)
        self.process_handler.wait_for_finish()
        self.reset = True

        answer = {'message': 'Resetting current queue', 'status': 'success'}
        return answer