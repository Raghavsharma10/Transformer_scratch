def get_outerframe_skip_importlib_frame(level):
        """
        There's a bug in Python3.4+, see http://bugs.python.org/issue23773,
        remove this and use sys._getframe(3) when bug is fixed
        """
        if sys.version_info < (3, 4):
            return sys._getframe(level)
        else:
            currentframe = inspect.currentframe()
            levelup = 0
            while levelup < level:
                currentframe = currentframe.f_back
                if currentframe.f_globals['__name__'] == 'importlib._bootstrap':
                    continue
                else:
                    levelup += 1
            return currentframe