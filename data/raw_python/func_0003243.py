async def shutdown(self):
        "Force stop the output stream, if there are more data to download, shutdown the connection"
        if self.stream:
            if not self.stream.dataeof and not self.stream.dataerror:
                self.stream.close(self.scheduler)
                await self.connection.shutdown()
            else:
                self.stream.close(self.scheduler)
            self.stream = None