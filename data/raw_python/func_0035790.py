def save_parsed_data_to_csv(self, output_filename='output.csv'):
        """ Outputs a csv file in accordance with parse_rectlabel_app_output method. This csv file is meant to accompany a set of pictures files
            in the creation of an Object Detection dataset.
            :param output_filename string, default makes sense, but for your convenience.
        """
        result = self.parse_rectlabel_app_output()

        ff = open(output_filename, 'w', encoding='utf8')

        for line in result:
            ff.write(line + '\n')

        ff.close()