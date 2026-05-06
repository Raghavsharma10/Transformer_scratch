def in_app() -> bool:
        """
        Judge where current working directory is in Django application or not.

        returns:
            - (Bool) cwd is in app dir returns True
        """
        try:
            MirageEnvironment.set_import_root()
            import apps
            if os.path.isfile("apps.py"):
                return True
            else:
                return False
        except ImportError:
            return False
        except:
            return False