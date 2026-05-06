def next_state(self):
        """This is a method that will be called when the time remaining ends.
        The current state can be: roasting, cooling, idle, sleeping, connecting,
        or unkown."""
        self.active_recipe_item += 1
        if self.active_recipe_item >= len(self.recipe):
            # we're done!
            return
        # show state step on screen
        print("--------------------------------------------")
        print("Setting next process step: %d" % self.active_recipe_item)
        print("time:%d, target: %ddegF, fan: %d, state: %s" %
              (self.recipe[self.active_recipe_item]['time_remaining'],
               self.recipe[self.active_recipe_item]['target_temp'],
               self.recipe[self.active_recipe_item]['fan_speed'],
               self.recipe[self.active_recipe_item]['state']
               ))
        print("--------------------------------------------")
        # set values for next state
        self.roaster.time_remaining = (
            self.recipe[self.active_recipe_item]['time_remaining'])
        self.roaster.target_temp = (
            self.recipe[self.active_recipe_item]['target_temp'])
        self.roaster.fan_speed = (
            self.recipe[self.active_recipe_item]['fan_speed'])
        # set state
        if(self.recipe[self.active_recipe_item]['state'] == 'roasting'):
            self.roaster.roast()
        elif(self.recipe[self.active_recipe_item]['state'] == 'cooling'):
            self.roaster.cool()
        elif(self.recipe[self.active_recipe_item]['state'] == 'idle'):
            self.roaster.idle()
        elif(self.recipe[self.active_recipe_item]['state'] == 'cooling'):
            self.roaster.sleep()