def list():
        """Return a list of all time zones known to the system."""
        handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        tzkey = winreg.OpenKey(handle, TZKEYNAME)
        result = [winreg.EnumKey(tzkey, i)
                  for i in range(winreg.QueryInfoKey(tzkey)[0])]
        tzkey.Close()
        handle.Close()
        return result