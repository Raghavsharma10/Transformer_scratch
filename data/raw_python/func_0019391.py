def get_postalcodes_around_radius(self, pc, radius):
        postalcodes = self.get(pc)
        if postalcodes is None:
            raise PostalCodeNotFoundException("Could not find postal code you're searching for.")
        else:
            pc = postalcodes[0]
        
        radius = float(radius)
        
        '''
        Bounding box calculations updated from pyzipcode
        '''        
        earth_radius  = 6371
        dlat = radius / earth_radius
        dlon = asin(sin(dlat) / cos(radians(pc.latitude)))
        lat_delta = degrees(dlat)
        lon_delta = degrees(dlon)
             
        if lat_delta < 0:
            lat_range = (pc.latitude + lat_delta, pc.latitude - lat_delta)
        else:
            lat_range = (pc.latitude - lat_delta, pc.latitude + lat_delta)
        
        long_range  = (pc.longitude - lat_delta, pc.longitude + lon_delta)    
        
        return format_result(self.conn_manager.query(PC_RANGE_QUERY % (
            long_range[0], long_range[1],
            lat_range[0], lat_range[1]
        )))