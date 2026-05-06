def trigger_frontend_build(self, event):
        """Event hook to trigger a new frontend build"""

        from hfos.database import instance
        install_frontend(instance=instance,
                         forcerebuild=event.force,
                         install=event.install,
                         development=self.development
                         )