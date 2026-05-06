def parse_rectlabel_app_output(self):
        """ Internal use mostly, finds all .json files in the current folder expecting them to all have been outputted by the RectLabel app
            parses each file returning finally an array representing a csv file where each element is a row and the 1st element [0] is the
            column headers.
            Could be useful for subsequent string manipulation therefore not prefixed with an underscore
            RectLabel info: https://rectlabel.com/
        """
        # get json files only
        files = []
        files = [f for f in os.listdir() if f[-5:] == '.json']

        if len(files) == 0:
            print('No json files found in this directory')
            return None

        max_boxes = 0        
        rows = []

        for each_file in files:
            f = open(each_file, 'r')
            j = f.read()            
            j = json.loads(j)            
            f.close()

            # running count of the # of boxes.
            if len(j['objects']) > max_boxes:
                max_boxes = len(j['objects'])

            # Each json file will end up being a row
            # set labels
            row = []

            for o in j['objects']:
                labels = {}
                labels['label'] = o['label']
                labels['x'] = o['x_y_w_h'][0]
                labels['y'] = o['x_y_w_h'][1]
                labels['width'] = o['x_y_w_h'][2]
                labels['height'] = o['x_y_w_h'][3]

                # String manipulation for csv
                labels_right_format = '\"' + json.dumps(labels).replace('"', '\"\"') + '\"'

                row.append(labels_right_format)

            row.insert(0, '\"' + j['filename'] + '\"')        

            rows.append(row)

        # one array element per row
        rows = [','.join(i) for i in rows]

        header = '\"image\"'
        
        for box_num in range(0, max_boxes):
            header += ', \"box\"' + str(box_num)

        rows.insert(0, header)
        return rows