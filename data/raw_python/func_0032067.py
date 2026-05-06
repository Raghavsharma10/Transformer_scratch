def getStore(self, name, domain):
        """Convenience method for the REPL. I got tired of typing this string every time I logged in."""
        return IRealm(self.original.store.parent).accountByAddress(name, domain).avatars.open()