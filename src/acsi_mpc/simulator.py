import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import os
import time
from mpc import MPC


class Simulator:

    def __init__(self, A, B, C, Q, R, RD, umin, umax, N):
        self.A = A
        self.B = B
        self.C = C
        self.num_outputs = C.shape[0]
        self.num_inputs = B.shape[1]

        if self.num_inputs == self.num_outputs == 1:
            self.mtype = 0
        
        else:
            self.mtype = 1

        self.mpc = MPC(A, B, C, Q, R, RD, umin, umax, N)


    def get_reference_trajectory(self, n, model_type=0):
        self.t = np.linspace(1, n, n)

        if model_type == 0:
            self.ref_traj = signal.square(self.t / 6)[np.newaxis, :]
        
        else:
            rx = 3 * np.ones(len(self.t))
            ry = np.ones(len(self.t))
            rz = signal.square(self.t / 32)
            ryaw = 2 * signal.square(self.t / 1000)

            self.ref_traj = np.row_stack((rx, ry, rz, ryaw))

    
    def simulate(self):

        U, X = self.establish_starting_state()

        for i in range(self.ref_traj.shape[1]):

            if i == 0:
                self.X_hist = X
                self.U_hist = U
                self.Y_hist = self.C @ X

            else:
                self.X_hist = np.column_stack((self.X_hist, X))
                self.U_hist = np.column_stack((self.U_hist, U))
                self.Y_hist = np.column_stack((self.Y_hist, self.C @ X))
            
            remaining_traj = self.ref_traj[:, i:]
            temp = time.time()
            U = self.mpc.get_control_input(X, U, remaining_traj)
            print(time.time()-temp)
            X = self.update_states(X, U)
    

    def establish_starting_state(self):
        U = np.zeros((self.B.shape[1], 1))
        X = np.zeros((self.A.shape[1], 1))

        return U, X

    def update_states(self, X, U):
        X = self.A @ X + self.B @ U
        
        return X

    def plot(self):
        self.t = self.t * .04
        if self.mtype == 0:
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('xkcd:black')

            ax.set_facecolor('xkcd:black')
            ax.tick_params(color='xkcd:white', labelcolor='xkcd:white')
            ax.spines['bottom'].set_color('xkcd:white')
            ax.spines['top'].set_color('xkcd:white')
            ax.spines['right'].set_color('xkcd:white')
            ax.spines['left'].set_color('xkcd:white')

            ax.plot(self.t, self.ref_traj[0, :])
            ax.plot(self.t, self.X_hist[0, :])

        else:
            figY, axY = plt.subplots(nrows=self.num_outputs)
            figU, axU = plt.subplots(nrows=self.num_outputs)
            figY.suptitle('Reference Tracking', color='xkcd:white', y=1.0)
            figU.suptitle('Control Usage', color='xkcd:white', y=1.0)
            figY.patch.set_facecolor('xkcd:black')
            figU.patch.set_facecolor('xkcd:black')

            Y_labels = [
                'X Position',
                'Y Position',
                'Z Position',
                'Yaw Position'
            ]

            U_labels = [
                'Roll',
                'Pitch',
                'Yaw',
                'Thrust'
            ]
            X_change = np.vstack((self.X_hist[1, :],self.X_hist[2, :],self.X_hist[0, :],self.X_hist[3, :]))
            for i in range(self.num_outputs):   
                axY[i].plot(self.t, self.ref_traj[i, :])
                axY[i].plot(self.t, X_change[i, :])
                axY[i].set_title(Y_labels[i], color='xkcd:white')

                axU[i].plot(self.t, self.U_hist[i, :])
                axU[i].set_title(U_labels[i], color='xkcd:white')

                axY[i].set_facecolor('xkcd:black')
                axU[i].set_facecolor('xkcd:black')

                axY[i].tick_params(color='xkcd:white', labelcolor='xkcd:white')
                axU[i].tick_params(color='xkcd:white', labelcolor='xkcd:white')

                axY[i].spines['bottom'].set_color('xkcd:white')
                axU[i].spines['bottom'].set_color('xkcd:white')

                axY[i].spines['top'].set_color('xkcd:white')
                axU[i].spines['top'].set_color('xkcd:white')

                axY[i].spines['right'].set_color('xkcd:white')
                axU[i].spines['right'].set_color('xkcd:white')

                axY[i].spines['left'].set_color('xkcd:white')
                axU[i].spines['left'].set_color('xkcd:white')

            figY.tight_layout()
            figU.tight_layout()

        plt.show()


def main(model_type=0):

    if model_type == 0: # SISO
        A = np.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        B = np.array([0.0, 1.0]).reshape(-1,1)
        C = np.array([1.0, 0.0]).reshape(1,-1)

        Q = np.array([1.0])
        R = np.array([0.1])
        RD = np.array([1.0])

        umin = np.array([-1.0])
        umax = np.array([1.0])

    else: # MIMO
        fname = 'Crazyflie_Model.mat'
        dirname = os.path.dirname(__file__)
        full_path = os.path.join(dirname, '../../models/'+fname)

        matfile = loadmat(full_path)

        A = matfile['A']
        B = matfile['B']
        C = matfile['C']

        Q = np.diag(np.array([1000., 10000., 1000., 1000.]))
        R = np.diag(np.array([1., 1., 1., 1e-8]))
        RD = np.diag(np.array([1000, 1000, 10, 1e-5]))

        umin = np.array([-.5, -.5, -.5, -10000.])[:, np.newaxis]
        umax = np.array([.5, .5, .5, 18000.])[:, np.newaxis]

    N = 30

    sim = Simulator(A, B, C, Q, R, RD, umin, umax, N)

    traj_length = 40*N

    sim.get_reference_trajectory(traj_length, model_type=model_type)

    sim.simulate()

    sim.plot()


if __name__ == '__main__':

    mtype = 1
    main(mtype)