def set_refresh_interval(self, seconds, **kwargs):
        """
        :param seconds:  -1 FOR NO REFRESH
        :param kwargs: ANY OTHER REQUEST PARAMETERS
        :return: None
        """
        if seconds <= 0:
            interval = -1
        else:
            interval = text_type(seconds) + "s"

        if self.cluster.version.startswith("0.90."):
            response = self.cluster.put(
                "/" + self.settings.index + "/_settings",
                data='{"index":{"refresh_interval":' + value2json(interval) + '}}',
                **kwargs
            )

            result = json2value(utf82unicode(response.all_content))
            if not result.ok:
                Log.error("Can not set refresh interval ({{error}})", {
                    "error": utf82unicode(response.all_content)
                })
        elif self.cluster.version.startswith(("1.4.", "1.5.", "1.6.", "1.7.", "5.", "6.")):
            result = self.cluster.put(
                "/" + self.settings.index + "/_settings",
                data={"index": {"refresh_interval": interval}},
                **kwargs
            )

            if not result.acknowledged:
                Log.error("Can not set refresh interval ({{error}})", {
                    "error": result
                })
        else:
            Log.error("Do not know how to handle ES version {{version}}", version=self.cluster.version)