def marksheet(self):
        """Returns an pandas empty dataframe object containing rows and columns for marking. This can then be passed to a google doc that is distributed to markers for editing with the mark for each section."""
        columns=['Number', 'Question', 'Correct (a fraction)', 'Max Mark', 'Comments']
        mark_sheet = pd.DataFrame() 
        for qu_number, question in enumerate(self.answers):
            part_no = 0
            for number, part in enumerate(question):
                if number>0:
                    if part[2] > 0:
                        part_no += 1
                        index = str(qu_number+1) +'_'+str(part_no)
                        frame = pd.DataFrame(columns=columns, index=[index])
                        frame.loc[index]['Number'] = index
                        frame.loc[index]['Question'] = part[0]
                        frame.loc[index]['Max Mark'] = part[2]
                        mark_sheet =  mark_sheet.append(frame)

        return mark_sheet.sort(columns='Number')