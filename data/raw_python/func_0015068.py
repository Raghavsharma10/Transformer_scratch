def power_source_type():
        """
        FreeBSD use sysctl hw.acpi.acline to tell if Mains (1) is used or Battery (0).
        Beware, that on a Desktop machines this hw.acpi.acline oid may not exist.
        @return: One of common.POWER_TYPE_*
        @raise: Runtime error if type of power source is not supported
        """
        try:
            supply=int(subprocess.check_output(["sysctl","-n","hw.acpi.acline"]))
        except:
            return common.POWER_TYPE_AC
 
        if supply == 1:
            return common.POWER_TYPE_AC
        elif supply == 0:
            return common.POWER_TYPE_BATTERY
        else:
            raise RuntimeError("Unknown power source type!")