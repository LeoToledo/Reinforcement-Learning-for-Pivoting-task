import numpy as np
import mujoco
from time import sleep
from scipy.spatial.transform import Rotation as R


class KukaImpedance():
    def __init__(self) -> None:        
        self.error_xpos = np.ones(6)*10
        self.error_xvel = np.ones(6)*10
        self.error_xacc = np.ones(6)*10

        self.K = np.eye(6)
        self.B = np.eye(6)

        self.K_null = np.eye(7)
        self.B_null = np.eye(7)
        self.q_nullspace = np.zeros(7)

        self.set_bk()

        self.tau_limits = np.array([87, 87, 87, 87, 12, 12, 12])
        

    def tau_impedance(self, mj, tau_g_tool=np.zeros(7,)):

        # if mj.tool_name != '':
        #     tau_g_tool = self.tool_grav_comp(mj)
        
        # H = self.get_inertia_matrix(sim)
        C = mj.get_coriolis_vector()
        J = mj.get_jacobian_site(site=mj.ee_site)

        # Compute generalized forces from a virtual external force.
        cartesian_acc_d = self.K.dot(self.error_xpos) + self.B.dot(self.error_xvel)

        # Add stiffness and damping in the null space of the the Jacobian
        # projection_matrix = np.eye(7) - J.T.dot(np.linalg.solve(J.dot(J.T), J))
        # print(np.linalg.inv(J.dot(J.T)))
        # Jpinv = J.T.dot(np.linalg.inv(J.dot(J.T)))
        Jpinv = np.linalg.pinv(J)
        # projection_matrix = np.eye(7) - J.T.dot(np.linalg.solve(J.dot(J.T), J))
        projection_matrix = np.eye(7) - Jpinv.dot(J)
        null_space_control = - self.B_null.dot(mj.get_robot_qvel()) + self.K_null.dot(self.q_nullspace - mj.get_robot_qpos())
        nullspace_torque_d = projection_matrix.dot(null_space_control)

        tau = J.T.dot(cartesian_acc_d) + C + tau_g_tool #+ nullspace_torque_d

        return tau
    

    def tool_grav_comp(self, mj):
        Jp_shape = (3, mj._nu)
        comp = np.zeros((mj._nu,))
        tool_mass = mj.sim.model.body_mass[mj.sim.model.body_names.index(mj.tool_name)]
        # for body, mass in [mj.tool_name, tool_mass] : #zip(self.name_bodies, self.mass_links):
        Jp = mj.sim.data.get_body_jacp(mj.tool_name).reshape(3, mj._nu)
        # Jr = mj.sim.data.get_body_jacr(mj.tool_name).reshape(3, mj._nu)
        comp = comp - np.dot(Jp.T, mj.sim.model.opt.gravity * tool_mass)# + np.dot(Jr.T, mj.sim.model.opt.gravity * tool_mass)
        return comp

    
    def calculate_errors(self, mj, xd, xdmat,
                         xvel_ref=np.zeros(6), 
                         xacc_ref=np.zeros(6)):
        
        # position error
        x, xmat = mj.get_robot_xpose()
        error_x = xd - x
        # error_x = np.zeros((3,))
        # print("\nerror_x = ", error_x)
    
        quat = mj.mat2quat(xmat)
        quatd = mj.mat2quat(xdmat)
        error_quat = self.get_quat_error(quat, quatd)

        # # orientation debugging
        # error_quat = np.zeros((3,))
        # error_quat = np.array([1, 0, 0])
        # print('\nquat = ', quat)
        # print('quatd = ', quatd)
        # print('error_quat = ', error_quat)
        # print('error_quat_norm = ', np.linalg.norm(error_quat))
        # print('product = ', quatd.dot(quat))

        self.error_xpos = np.concatenate((error_x, error_quat))

        # velocity error
        xvel_act = mj.get_robot_xvel()
        self.error_xvel = xvel_ref - xvel_act

        if mj.render and mj.render_cartesian_frames:
            # actual
            mj.render_cartesian_frame(x, xmat, alpha=0.2)
            
            # desired
            mj.render_cartesian_frame(xd, xdmat)

            # world frame
            mj.render_cartesian_frame(np.zeros((3,)), np.eye(3))
    

    def set_bk(self, b=100, k=1000, b_rot=10, k_rot=100) -> None:
        self._set_k()
        self._set_b()


    def _set_b(self, b=0, b_rot=0):
        if b == 0 and b_rot == 0:
            self.B[:3,:3] = 2 * 0.707 * np.sqrt(self.K[:3,:3])
            self.B[3:,3:] = 2 * 0.707* np.sqrt(self.K[3:,3:])
        else:
            self.B[:3,:3] = np.eye(self.B[:3,:3].shape[0])*b
            self.B[3:,3:] = np.eye(self.B[3:,3:].shape[0])*b_rot


    def _set_k(self, k=0, k_rot=0):
        if k==0 and k_rot == 0:
            self.K[:3,:3] = np.eye(3)*1000
            self.K[3:,3:] = np.eye(3)*60
        else:
            self.K[:3,:3] = np.eye(self.K[:3,:3].shape[0])*k
            self.K[3:,3:] = np.eye(self.K[3:,3:].shape[0])*k_rot


    def get_quat_error(self, quat, quatd):
        # for some reasons, the methods commented do not work

        # # method 1
        # if qd.dot(q) < 0:
        #     q = -q
        # q_error = qd[0] * q[1:] - q[0] * qd[1:] - skew(qd[1:]).dot(q[1:])

        # # method 2
        # q_error = np.zeros((3,))
        # mujoco.functions.mju_subQuat(q_error, qd, q)

        q_error = np.zeros((4,))
        res = np.zeros((3,))
        q_t = np.zeros((4,))
        mujoco.mju_negQuat(q_t, quat)
        mujoco.mju_mulQuat(q_error, quatd, q_t)
        mujoco.mju_quat2Vel(res, q_error, 1)

        return res

    def move_to_point(self, mj, xd=np.zeros((3,)), xdmat=np.zeros((3,3)), eps=0.01, eps_ori=0.1, stay_there=False):
        x, xmat = mj.get_robot_xpose()

        if np.all(xd) == 0:
            xd = x
        if np.all(xdmat) == 0:
            xdmat = xmat
        
        self.calculate_errors(mj, xd=xd, xdmat=xdmat)
        # mj.set_object_xpos()
        
        while np.linalg.norm(self.error_xpos[:3]) > eps or np.linalg.norm(self.error_xpos[3:]) > eps_ori:
        # while True:
        
            self.calculate_errors(mj, xd=xd, xdmat=xdmat)
            
            u = self.tau_impedance(mj)

            mj.set_joint_torques(u)

            mj.step_sim()
            # sleep(mj.sim.model.opt.timestep)
