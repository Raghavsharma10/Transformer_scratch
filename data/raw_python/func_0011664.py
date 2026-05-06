def stop(self):
        """
        Stop animation thread.
        """
        time.sleep(self.speed)
        self._count = -9999
        sys.stdout.write(self.reverser + '\r\033[K\033[A')
        sys.stdout.flush()
        return