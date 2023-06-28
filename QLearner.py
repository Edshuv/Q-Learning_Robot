""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Eduard Shuvaev (replace with your name)  		  	   		  	  			  		 			     			  	 
GT User ID: eshuvaev3 (replace with your User ID)  		  	   		  	  			  		 			     			  	 
GT ID: 903362621 (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import random as rand  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		  	  		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		  	  		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		  	  		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		  	  		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		  	  		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		  	  		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  	  		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  		  	   		  	  		  		  		    	 		 		   		 		  
        self.s = 0  		  	   		  	  		  		  		    	 		 		   		 		  
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        if self.dyna != 0:
            # self.Tc = np.full((self.num_states, self.num_actions, self.num_states), 0.00001)
            self.Tc = np.zeros((self.num_states, self.num_actions, self.num_states))
            self.R = np.zeros((self.num_states, self.num_actions))
            self.trS = []
            self.trA = []

    #   create a Q[s,a] table
        self.q = np.zeros((self.num_states, self.num_actions))
  		  	   		  	  		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.s = s

        # Generate a random number between 0 and 1
        rnd = np.random.random()
        # rnd = rand.random()

        # If random number < epsilon, take a random action
        if rnd < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        # Else, take the action with the highest value in the current state
        else:
            action = np.argmax(self.q[s])

        if self.verbose:
            print(f"s = {s}, a = {action}")  		  	   		  	  		  		  		    	 		 		   		 		  
        return action  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		  	  		  		  		    	 		 		   		 		  
        :type r: float  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """

        # Update Q(s,a)
        # Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmax_a'(Q[s', a'])])
        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.q[s_prime]))

        if self.dyna != 0:
            self.Tc[self.s, self.a, s_prime] += 1
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r
            self.trS.append(self.s)
            self.trA.append(self.a)
            for i in range(self.dyna):
                self.hallucination()


        # Generate a random number between 0 and 1
        rnd = np.random.random()
        # rnd = rand.random()

        # If random number < epsilon, take a random action
        if rnd < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        # Else, take the action with the highest value in the current state
        else:
            action = np.argmax(self.q[s_prime])

        # update rar: rar = rar * radr
        self.rar = self.rar * self.radr

        # update the state and action
        self.s = s_prime
        self.a = action

        if self.verbose:  		  	   		  	  		  		  		    	 		 		   		 		  
            print(f"s = {s_prime}, a = {action}, r={r}")

        return action

    def hallucination (self):
        s = rand.choice(self.trS)
        a = rand.choice(self.trA)
        # s = np.random.Generator.choice(self.trS)
        # a = np.random.Generator.choice(self.trA)
        h_prime = np.argmax(self.Tc[s,a])
        # if not self.Tc[s,a].any():
        #     h_prime = 0
        # else:
        #     h_prime = rand.choice(np.where(self.Tc[s,a] != 0))
        # print("PRINT: ", np.where(tarray > 0.00001))
        # exit(1)
        # h_prime = rand.choice(np.where(tarray > 0))
        # h_prime = np.argmax(self.Tc[s,a]/np.sum(self.Tc[s,a]))
        # print(self.trS)
        # print(s)
        # print(self.Tc[s,a])
        # print(h_prime)# let's find the best next state
        h_r = self.R[s,a]                       # grub reward from randomly chosen states
        # update hullucinatede Q
        self.q[s,a] = (1 - self.alpha) * self.q[s,a] + self.alpha * (h_r + self.gamma * np.max(self.q[h_prime]))





    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "eshuvaev3"  # replace tb34 with your Georgia Tech username.

if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		  	  		  		  		    	 		 		   		 		  
