# modified by chuhaof2

from __future__ import print_function

from tabulate import tabulate
import numpy as np
import pdb


class HMM(object):

    def __init__(self, A, B, pi0=None, states=None, emissions=None):
        """
        :param A: Transition matrix of shape (n, n) (n = number of states)
        :param B: Emission matrix of shape (n, b) (b = number of outputs)
        :param pi0: Initial State Probability vector of size n, leave blank for uniform probabilities
        :param states: State names/labels as list
        :param emissions: Emission names/labels as list
        """
        self.A = A
        self.B = B
        self.n_states = A.shape[0]
        self.n_emissions = B.shape[1]
        self.states = states
        self.emissions = emissions
        self.pi0 = pi0

        if pi0 is None:
            self.pi0 = np.full(self.n_states, 1.0 / self.n_states)

        if states is None:
            self.states = [chr(ord('A') + i) for i in range(self.n_states)]

        if emissions is None:
            self.emissions = [str(i) for i in range(self.n_emissions)]

    def print_matrix(self, M, headers=None, flag=None): # flag for matrix type, 0 for Alpha, 1 for Beta, 2 for Gamma
        """
        Print matrix in tabular form
        :param M: Matrix to print
        :param headers: Optional headers for columns, default is state names
        :return: tabulated encoding of input matrix
        """
        headers = headers or self.states

        if M.ndim > 1:        # add additional formating for partial credit
            if flag == 0:
                headers = ['Alpha'] + headers
            elif flag == 1:
                headers = ['Beta'] + headers
            elif flag == 2:
                headers = ['Gamma'] + headers
            else:
                headers = [' '] + headers
            
            data = [['t={}'.format(i + 1)] + [j for j in row] for i, row in enumerate(M)]
        else:
            data = [[j for j in M]]
        print(tabulate(data, headers, tablefmt="grid", numalign="right"))
        return None

    def forward_algorithm(self, seq):
        """
        Apply forward algorithm to calculate probabilities of seq
        :param seq: Observed sequence to calculate probabilities upon
        :return: Alpha matrix with 1 row per time step
        """
        
        T = len(seq)

        # Initialize forward probabilities matrix Alpha
        Alpha = np.zeros((T, self.n_states))

        # Your implementation here
        
        # initialize the first row    
        Alpha[0,:] = self.pi0*self.B[:,seq[0]] / np.sum(self.pi0*self.B[:,seq[0]])
        
        # calculate the remaining rows
        for i in range(1,T):
            Alpha[i,:] = self.B[:,seq[i]] * np.dot(np.transpose(self.A), np.transpose(Alpha[i-1,:]))
            Alpha[i,:] = Alpha[i,:] / np.sum(Alpha[i,:])
                
        return Alpha    

    def backward_algorithm(self, seq):
        """
        Apply backward algorithm to calculate probabilities of seq
        :param seq: Observed sequence to calculate probabilities upon
        :return: Beta matrix with 1 row per timestep
        """

        T = len(seq)

        # Initialize backward probabilities matrix Beta
        Beta = np.zeros((T, self.n_states))

        # Your implementation here
            
        # initialize the last row
        Beta[T-1,:] = np.ones(self.n_states)
        
        # calculate the remaining rows
        for i in range(T-1, 0, -1):
            Beta[i-1,:] = np.dot(self.A, np.transpose(self.B[:,seq[i]] * Beta[i,:]))
            
        return Beta

    def forward_backward(self, seq):
        """
        Applies forward-backward algorithm to seq
        :param seq: Observed sequence to calculate probabilities upon
        :return: Gamma matrix containing state probabilities for each timestamp
        :raises: ValueError on bad sequence
        """

        # Convert sequence to integers
        if all(isinstance(i, str) for i in seq):
            seq = [self.emissions.index(i) for i in seq]

        # Infer time steps
        T = len(seq)
        
        # Calculate forward probabilities matrix Alpha
        Alpha = self.forward_algorithm(seq)
        # Initialize backward probabilities matrix Beta
        Beta = self.backward_algorithm(seq)

        # Initialize Gamma matrix
        Gamma = np.zeros((T, self.n_states))
        
        # Your implementation here
        
        for i in range(T):
            Gamma[i,:] = Alpha[i,:] * Beta[i,:]
            Gamma[i,:] = Gamma[i,:] / np.sum(Gamma[i,:])

#         return Alpha, Beta, Gamma          # just for partial credit
        return Gamma