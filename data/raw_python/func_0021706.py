def write_text_rows(writer, rows):
    '''Write CSV row data which may include text.'''
    for row in rows:
        try:
            writer.writerow(row)
        except UnicodeEncodeError:
            # Python 2 csv does badly with unicode outside of ASCII
            new_row = []
            for item in row:
                if isinstance(item, text_type):
                    new_row.append(item.encode('utf-8'))
                else:
                    new_row.append(item)
            writer.writerow(new_row)