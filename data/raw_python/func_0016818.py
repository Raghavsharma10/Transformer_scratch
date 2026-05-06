def updated_dimensions(self):
        """ Inform montblanc about dimension sizes """
        return [("ntime", args.ntime),      # Timesteps
                ("nchan", args.nchan),      # Channels
                ("na", args.na),            # Antenna
                ("npsrc", len(lm_coords))]