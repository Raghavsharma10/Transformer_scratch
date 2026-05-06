def write(self, bucket, rows):
        """https://github.com/frictionlessdata/tableschema-pandas-py#storage
        """

        # Prepare
        descriptor = self.describe(bucket)
        new_data_frame = self.__mapper.convert_descriptor_and_rows(descriptor, rows)

        # Just set new DataFrame if current is empty
        if self.__dataframes[bucket].size == 0:
            self.__dataframes[bucket] = new_data_frame

        # Append new data frame to the old one setting new data frame
        # containing data from both old and new data frames
        else:
            self.__dataframes[bucket] = pd.concat([
                self.__dataframes[bucket],
                new_data_frame,
            ])