def _add_intent_interactive(self, intent_num=0):        
        '''
        Interactively add a new intent to the intent schema object 
        '''
        print ("Name of intent number : ", intent_num)
        slot_type_mappings = load_builtin_slots()
        intent_name = read_from_user(str)
        print ("How many slots?")        
        num_slots = read_from_user(int)
        slot_list = []
        for i in range(num_slots):
            print ("Slot name no.", i+1)
            slot_name = read_from_user(str).strip()
            print ("Slot type? Enter a number for AMAZON supported types below,"
                   "else enter a string for a Custom Slot")
            print (json.dumps(slot_type_mappings, indent=True))
            slot_type_str = read_from_user(str)
            try: slot_type = slot_type_mappings[int(slot_type_str)]['name'] 
            except: slot_type = slot_type_str
            slot_list += [self.build_slot(slot_name, slot_type)]                    
        self.add_intent(intent_name, slot_list)