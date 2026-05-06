def get_hardware_source_by_id(self, hardware_source_id: str, version: str):
        """Return the hardware source API matching the hardware_source_id and version.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Hardware API requested version %s is greater than %s." % (version, actual_version))
        hardware_source = HardwareSourceModule.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
        return HardwareSource(hardware_source) if hardware_source else None