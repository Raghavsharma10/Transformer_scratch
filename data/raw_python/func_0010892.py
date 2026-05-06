def get_coords(self):
        """Utility function to get the coordinates as a single list of tuples."""
        out_list = []
        for i in range(len(self.x_coord_list)):
            out_list.append((self.x_coord_list[i],self.y_coord_list[i],self.z_coord_list[i],))
        return out_list