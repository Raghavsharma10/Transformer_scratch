def _get_description(self, args: Tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return the dictionary to be sent to the queue."""
        return {
            'id': uuid1().hex,
            'args': args,
            'kwargs': kwargs,
            'module': self._module_name,
            'function': self.f.__name__,
            'sender_hostname': socket.gethostname(),
            'sender_pid': os.getpid(),
            'sender_cmd': ' '.join(sys.argv),
            'sender_timestamp': datetime.utcnow().isoformat()[:19],
        }