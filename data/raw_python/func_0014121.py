def set_attribute(self, attr, value = True):
    """ Sets a custom attribute for our Webkit instance. Possible attributes are:

      * ``auto_load_images``
      * ``dns_prefetch_enabled``
      * ``plugins_enabled``
      * ``private_browsing_enabled``
      * ``javascript_can_open_windows``
      * ``javascript_can_access_clipboard``
      * ``offline_storage_database_enabled``
      * ``offline_web_application_cache_enabled``
      * ``local_storage_enabled``
      * ``local_storage_database_enabled``
      * ``local_content_can_access_remote_urls``
      * ``local_content_can_access_file_urls``
      * ``accelerated_compositing_enabled``
      * ``site_specific_quirks_enabled``

    For all those options, ``value`` must be a boolean. You can find more
    information about these options `in the QT docs
    <http://developer.qt.nokia.com/doc/qt-4.8/qwebsettings.html#WebAttribute-enum>`_.
    """
    value = "true" if value else "false"
    self.conn.issue_command("SetAttribute",
                            self._normalize_attr(attr),
                            value)