def check_label(self):
        """ Check label for target and fix if necessary.

        Forcing the target name to Saturn here, because some observations of
        the rings have moons as a target, but then the standard map projection
        onto the Saturn ring plane fails.

        See also
        --------
        https://isis.astrogeology.usgs.gov/IsisSupport/index.php/topic,3922.0.html
        """
        if not self.is_ring_data:
            return
        targetname = getkey(
            from_=self.pm.raw_cub, grp="instrument", keyword="targetname"
        )

        if targetname.lower() != "saturn":
            editlab(
                from_=self.pm.raw_cub,
                options="modkey",
                keyword="TargetName",
                value="Saturn",
                grpname="Instrument",
            )