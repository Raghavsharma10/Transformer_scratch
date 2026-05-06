def _handle_signals(self):
        """
        因为主进程的reactor重新处理了SIGINT，会导致子进程也会响应，改为SIG_IGN之后，就可以保证父进程先退出，之后再由父进程term所有的子进程
        :return:
        """
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_IGN)