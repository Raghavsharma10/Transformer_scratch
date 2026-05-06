def extract_tree_block(self):
        "iterate through data file to extract trees"        

        lines = iter(self.data)
        while 1:
            try:
                line = next(lines).strip()
            except StopIteration:
                break
    
            # enter trees block
            if line.lower() == "begin trees;":
                while 1:
                    # iter through trees block
                    sub = next(lines).strip().split()
                    
                    # skip if a blank line
                    if not sub:
                        continue

                    # look for translation
                    if sub[0].lower() == "translate":
                        while sub[0] != ";":
                            sub = next(lines).strip().split()
                            self.tdict[sub[0]] = sub[-1].strip(",")

                    # parse tree blocks
                    if sub[0].lower().startswith("tree"):
                        self.newicks.append(sub[-1])
        
                    # end of trees block
                    if sub[0].lower() == "end;":
                        break