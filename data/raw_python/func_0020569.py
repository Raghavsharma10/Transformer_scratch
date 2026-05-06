def get_error_message(self):
        """
        Return an error message based on atomic-reactor's metadata
        """
        error_reason = self.get_error_reason()
        if error_reason:
            error_message = error_reason.get('pod') or None
            if error_message:
                return "Error in pod: %s" % error_message
            plugin = error_reason.get('plugin')[0] or None
            error_message = error_reason.get('plugin')[1] or None
            if error_message:
                # Plugin has non-empty error description
                return "Error in plugin %s: %s" % (plugin, error_message)
            else:
                return "Error in plugin %s" % plugin