def get_volume(self, controller, zone):
        """ Gets the volume level which needs to be doubled to get it to the range of 0..100 -
        it is located on a 2 byte offset """
        volume_level = self.get_zone_info(controller, zone, 2)
        if volume_level is not None:
            volume_level *= 2
        return volume_level