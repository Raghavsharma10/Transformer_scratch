def thread_with_callback(on_error, on_done, requete_with_callback):
    """
    Return a thread emiting `state_changed` between each sub-requests.

    :param on_error: callback str -> None
    :param on_done: callback object -> None
    :param requete_with_callback: Job to execute. monitor_callable -> None
    :return: Non started thread
    """

    class C(THREAD):

        error = SIGNAL(str)
        done = SIGNAL(object)
        state_changed = SIGNAL(int, int)

        def __del__(self):
            self.wait()

        def run(self):
            try:
                r = requete_with_callback(self.state_changed.emit)
            except (ConnexionError, StructureError) as e:
                self.error.emit(str(e))
            else:
                self.done.emit(r)

    th = C()
    th.error.connect(on_error)
    th.done.connect(on_done)
    return th