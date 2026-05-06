def halt(self):
        """Shutdown the drone.

        This method does not land or halt the actual drone, but the
        communication with the drone. You should call it at the end of your
        application to close all sockets, pipes, processes and threads related
        with this object.
        """
        with self.lock:
            self.com_watchdog_timer.cancel()
            self.ipc_thread.stop()
            self.ipc_thread.join()
            self.network_process.terminate()
            self.network_process.join()