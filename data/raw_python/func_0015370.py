def set_geo_area(self, area):
        '''Sets the geo area for the map.

        * africa
        * asia
        * europe
        * middle_east
        * south_america
        * usa
        * world
        '''
        
        if area in self.__areas:
            self.geo_area = area
        else:
            raise UnknownChartType('Unknown chart type for maps: %s' %area)