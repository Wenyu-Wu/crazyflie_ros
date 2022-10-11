#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from cvxopt import matrix, solvers


class MPC:

    def __init__(self, A, B, C, Q, R, RD, umin, umax, N):
        solvers.options['show_progress'] = False

        self.N = N # horizon length
        B, C = self.dimension_fill(B, C)

        self.num_states = A.shape[1]
        self.num_inputs = B.shape[1]
        self.num_outputs = C.shape[0]

        self.precompute(A, B, C, Q, R, RD, umin, umax)

    
    def dimension_fill(self, B, C):
        if B.ndim < 2:
            B = B[:, np.newaxis]

        if C.ndim < 2:
            C = C[np.newaxis, :]
        
        return B, C

    
    def precompute(self, A, B, C, Q, R, RD, umin, umax):

        Qbar, Rbar, RbarD = self.build_bars(Q, R, RD, self.N)

        Sx = self.build_Sx(A, C, self.N)

        Su = self.build_Su(A, B, C, self.N)

        L = self.build_L(self.N)

        self.G = self.build_G(L, self.N)
        
        self.P = self.build_P(Rbar, RbarD, Qbar, Su, L)

        self.Fu1, self.Fu2, self.Fr, self.Fx = self.build_Fs(Rbar, Qbar, Su, Sx, L)
        
        self.W0 = self.build_W0(self.N, umin, umax)

        self.S = self.build_S(self.N)
    

    def build_bars(self, Q, R, RD, N):
        Qbar = np.kron(np.eye(N), Q) # weight penalty for states
        Rbar = np.kron(np.eye(N), R) # weight penalty for control effort
        RbarD = np.kron(np.eye(N), RD) # weight penalty for change in control effort

        assert Qbar.shape == (self.num_outputs * N, self.num_outputs * N)
        assert Rbar.shape == (self.num_inputs * N, self.num_inputs * N)
        assert RbarD.shape == (self.num_inputs * N, self.num_inputs * N)

        return Qbar, Rbar, RbarD


    def build_Sx(self, A, C, N):
        '''
            Sx is how the initial state propogates to the future predicted outputs
            [ C * A
              C * A^2
                :
              C * A^N ]
        '''
        # This commented code is a more intuitive but slower way to build Sx
        # Sx1 = C @ A
        # for i in range(2, N+1):
        #     next_element = C @ np.linalg.matrix_power(A, i)
        #     Sx1 = np.row_stack((Sx1, next_element))

        Sx = np.array([
            C @ np.linalg.matrix_power(A, i)
            for i in range(1, N+1)
        ]).reshape(self.num_outputs * N, self.num_states)

        assert Sx.shape == (self.num_outputs * N, self.num_states)

        return Sx
    

    def build_Su(self, A, B, C, N):
        '''
            S is how the history of control inputs propogate to the future predicted outputs
        '''

        Su1 = np.array([
            C @ np.linalg.matrix_power(A, i) @ B
            for i in range(N)
        ]).reshape(self.num_outputs * N, self.num_inputs)

        assert Su1.shape == (self.num_outputs * N, self.num_inputs)
        
        # This commented code is a more intuitive but slower way to build Su2
        # zero_array = np.zeros((self.num_outputs, self.num_inputs))
        # Su2 = np.concatenate((zero_array, Su1[:-4,:]))
        # for i in range(2, N):
        #     column_top = np.tile(zero_array, (i, 1))
        #     next_column = np.concatenate((column_top, Su1[:-i*self.num_outputs,:]))
        #     Su2 = np.column_stack((Su2, next_column))

        Su2 = np.array([
            np.concatenate((np.tile(np.zeros((self.num_outputs, self.num_inputs)),(i,1)), Su1[:-i*self.num_outputs,:])).T
            for i in range(1, N)
        ]).reshape(self.num_inputs * (N - 1), self.num_outputs * N).T

        assert Su2.shape == (self.num_outputs * N, self.num_inputs * (N - 1))
        assert np.allclose(Su2[self.num_outputs:, :self.num_inputs], Su1[:-self.num_outputs, :])
        assert np.allclose(Su2[-self.num_outputs:, -self.num_inputs:], Su1[:self.num_outputs, :])

        Su = np.column_stack((Su1, Su2))

        assert Su.shape == (self.num_outputs * N, self.num_inputs * N)

        return Su

    
    def build_L(self, N):
        L = np.tril(np.tile(np.eye(self.num_inputs), (N, N)))

        assert L.shape == (self.num_inputs * N, self.num_inputs * N)

        return L

    
    def build_G(self, L, N):
        G = np.row_stack((L, np.negative(L)))

        assert G.shape == (2 * self.num_inputs * N, self.num_inputs * N)

        return G

    
    def build_W0(self, N, umin, umax):
        umin_arr = np.tile(np.negative(umin), (N, 1))
        umax_arr = np.tile(umax, (N, 1))

        W0 = np.row_stack((umax_arr, umin_arr))

        assert W0.shape == (2 * self.num_inputs * N, 1)

        return W0

    
    def build_S(self, N):
        S = np.zeros((2 * self.num_inputs * N, self.num_states))

        assert S.shape == (2 * self.num_inputs * N, self.num_states)

        return S


    def build_P(self, Rbar, RbarD, Qbar, Su, L):
        P = 2 * (Su.T @ Qbar @ Su + L.T @ Rbar @ L + RbarD)

        assert P.shape == (self.num_inputs * self.N, self.num_inputs * self.N)

        return P

    
    def build_Fs(self, Rbar, Qbar, Su, Sx, L):
        Fu1 = 2 * (Rbar.T @ L.T)
        Fu2 = 2 * (Su[:, :self.num_inputs].T @ Qbar @ Su).T
        Fr = -2 * (Su.T @ Qbar.T)
        Fx = 2 * (Su.T @ Qbar.T @ Sx)

        assert Fu1.shape == (self.num_inputs * self.N, self.num_inputs * self.N)
        assert Fu2.shape == (self.num_inputs * self.N, self.num_inputs)
        assert Fr.shape == (self.num_inputs * self.N, self.num_outputs * self.N)
        assert Fx.shape == (self.num_inputs * self.N, self.num_states)

        return Fu1, Fu2, Fr, Fx


    def calculate_q(self, X, U, traj_horizon):
        q = self.Fx @ X \
            + self.Fu2 @ U \
            + self.Fu1 @ np.tile(U, (self.N, 1)) \
            + self.Fr @ traj_horizon

        assert q.shape == (self.num_inputs * self.N, 1)

        return q


    def calculate_W(self, U):
        U_neg = np.tile(np.negative(U), (self.N, 1))
        U_pos = np.tile(U, (self.N, 1))
        lastU = np.row_stack((U_neg, U_pos))

        W = self.W0 + lastU

        assert W.shape == (2 * self.num_inputs * self.N, 1)

        return W


    def calculate_h(self, W, X):
        h = W + self.S @ X

        return h

    
    def get_control_input(self, X, U, traj):
        U, traj = self.fix_dimensions(U, traj)

        traj_horizon = self.get_trajectory_horizon(traj)

        q = self.calculate_q(X, U, traj_horizon)

        W = self.calculate_W(U)

        h = self.calculate_h(W, X)

        sol = solvers.qp(matrix(self.P), matrix(q), matrix(self.G), matrix(h))

        future_control = np.array(sol['x'])

        U += future_control[:self.num_outputs]
        #print(U.shape)
        return U

    
    def fix_dimensions(self, U, traj):
        # dimension fill if single input system
        if U.ndim < 2:
            U = U[:, np.newaxis]
        
        # dimension fill if trajectory is 1D
        if traj.ndim < 2:
            traj = traj[np.newaxis, :]
        
        # ensure the trajectories for each input lie on a row
        if traj.shape[1] == self.num_inputs:
            traj = traj.T

        return U, traj


    def get_trajectory_horizon(self, traj):
        # if horizon is greater than trajectory length, increase trajectory by stacking the last position

        while self.N > traj.shape[1]:
            traj = np.column_stack((traj,traj[:,-1]))

        traj_horizon = traj[:, :self.N]
        traj_horizon = self.stack_trajectories(traj_horizon)

        return traj_horizon
    

    def stack_trajectories(self, traj):
        traj = traj.T.reshape(-1,1)
        
        return traj