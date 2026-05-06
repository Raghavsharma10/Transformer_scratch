def generate_simhash(self, item):
        """
        Generate simhash based on title, description, keywords, p_texts and links_text.
        """
        list = item['p_texts'] + item['links_text']
        list.append(item['title'])
        list.append(item['description'])
        list.append(item['keywords'])
        return Simhash(','.join(list).strip()).hash