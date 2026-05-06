def get_response(self):
        """
            Get the response from the server. *This may not return the full response*

        :return: Response data
        """
        while not self.comm_chan.recv_ready():
            time.sleep(0.5)
        return self.comm_chan.recv(2048)