def respond_client(self, answer, socket):
        """Send an answer to the client."""
        response = pickle.dumps(answer, -1)
        socket.sendall(response)
        self.read_list.remove(socket)
        socket.close()