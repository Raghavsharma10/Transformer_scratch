def output_sector_csv(self,csv_path,file_dict_key,out_path):
        """
        Segment forecast tracks to only output data contined within a 
        region in the CONUS, as defined by the mapfile.

        Args:
            csv_path(str): Path to the full CONUS csv file.
            file_dict_key(str): Dictionary key for the csv files, 
                currently either 'track_step' or 'track_total'
            out_path (str): Path to output new segmented csv files.
        Returns:
            Segmented forecast tracks in a csv file.
        """
        csv_file = csv_path + "{0}_{1}_{2}_{3}.csv".format(
                                                        file_dict_key,
                                                        self.ensemble_name,
                                                        self.member,
                                                        self.run_date.strftime(self.date_format))
        if exists(csv_file):
            csv_data = pd.read_csv(csv_file)
            
            if self.inds is None:
                lon_obj = csv_data.loc[:,"Centroid_Lon"]
                lat_obj = csv_data.loc[:,"Centroid_Lat"]
            
                self.inds = np.where((self.ne_lat>=lat_obj)&(self.sw_lat<=lat_obj)\
                        &(self.ne_lon>=lon_obj)&(self.sw_lon<=lon_obj))[0]
            
            if np.shape(self.inds)[0] > 0:
                csv_data = csv_data.reindex(np.array(self.inds)) 
                sector_csv_filename = out_path + "{0}_{1}_{2}_{3}.csv".format(
                                                        file_dict_key,
                                                        self.ensemble_name,
                                                        self.member,
                                                        self.run_date.strftime(self.date_format))
                print("Output sector csv file " + sector_csv_filename)
                csv_data.to_csv(sector_csv_filename,
                        na_rep="nan",
                        float_format="%0.5f",
                        index=False)
                os.chmod(sector_csv_filename, 0o666)
            else:
                print('No {0} {1} sector data found'.format(self.member,
                                self.run_date.strftime("%Y%m%d")))
            
        else:
            print('No {0} {1} csv file found'.format(self.member,
                                self.run_date.strftime("%Y%m%d")))
        return