def _parse_bounding_box(bounding_box):
        '''
        Parse response bounding box from the CapakeyRestGateway to (MinimumX, MinimumY, MaximumX, MaximumY)
        
        :param bounding_box: response bounding box from the CapakeyRestGateway
        :return: (MinimumX, MinimumY, MaximumX, MaximumY)
        '''
        coordinates = json.loads(bounding_box)["coordinates"]
        x_coords = [x for x, y in coordinates[0]]
        y_coords = [y for x, y in coordinates[0]]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)