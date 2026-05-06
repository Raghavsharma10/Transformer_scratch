def by(self, technology):
        """
        Get the plugins registered in PedalPi by technology

        :param PluginTechnology technology: PluginTechnology identifier
        """
        if technology == PluginTechnology.LV2 \
        or str(technology).upper() == PluginTechnology.LV2.value.upper():
            return self.lv2_builder.all
        else:
            return []