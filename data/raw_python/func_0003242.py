def close(self):
        "Stop the output stream, but further download will still perform"
        if self.stream:
            self.stream.close(self.scheduler)
            self.stream = None