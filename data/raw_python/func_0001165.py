def incoming_messages(self) -> t.List[t.Tuple[float, bytes]]:
        """Consume the receive buffer and return the messages.

        If there are new messages added to the queue while this funciton is being
        processed, they will not be returned. This ensures that this terminates in
        a timely manner.
        """
        approximate_messages = self._receive_buffer.qsize()
        messages = []
        for _ in range(approximate_messages):
            try:
                messages.append(self._receive_buffer.get_nowait())
            except queue.Empty:
                break
        return messages