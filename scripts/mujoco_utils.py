import numpy as np
import mujoco as mj
import mujoco_viewer
from copy import deepcopy

class MujocoUtilsCartesian():
    def __init__(self, render=True, render_cartesian_frames=True, ee_site_name='', tool_name='') -> None:
        # old
        # self.sim = None
        self.viewer = None
        
        # new
        self.model = None
        self.data = None

        self.ee_site = ee_site_name
        self.tool_name = tool_name
        self._nu = 7
        self.render = render
        self.render_cartesian_frames = render_cartesian_frames
        self.xtgt = np.zeros(3)

    
    def load_model_mujoco(self, path_to_model):
        self.model = mj.MjModel.from_xml_path(path_to_model)  # mujoco_panda/models/franka_panda_no_gripper"
        self.data = mj.MjData(self.model)

        if self.render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)


    def get_jacobian_site(self, site, dof=3, nu=7):
        Jp_shape = (dof, nu)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site))
        # if self.tool_name != '':
        #     nu_free = 6
        #     jacp = np.concatenate([jacp[:, 0:nu], jacp[:, nu+nu_free:2*nu+nu_free], jacp[:, 2*(nu+nu_free):2*(nu+nu_free)+nu]])#.reshape(Jp_shape)
        #     jacr = np.concatenate([jacr[:, 0:nu], jacr[:, nu+nu_free:2*nu+nu_free], jacr[:, 2*(nu+nu_free):2*(nu+nu_free)+nu]])#.reshape(Jp_shape)
        return np.vstack((jacp[:, :nu], jacr[:, :7]))
    
    def get_robot_qpos(self):
        return self.data.qpos[0:self._nu]

    def get_robot_qvel(self):
        return self.data.qvel[0:self._nu]

    def set_joint_torques(self, tau):
        self.data.ctrl[:self._nu] = tau

    
    def mat2quat(self, xmat):
        quat = np.zeros((4,))
        mj.mju_mat2Quat(quat, xmat.flatten())
        return quat

    def get_time(self):
        return self.data.time

    def get_site_xpose(self, site_name):
        x = deepcopy(self.data.site(site_name).xpos)
        xmat = deepcopy(self.data.site(site_name).xmat.reshape(3,3))
        return x, xmat

    def get_site_quat_from_mat(self, site_name):
        xmat = self.data.get_site_xmat(site_name)
        xquat = np.zeros((4,))
        mj.functions.mju_mat2Quat(xquat, xmat.flatten())
        return xquat
    
    def get_robot_xpose(self):
        return self.get_site_xpose(self.ee_site)

    def get_object_xpose(self):
        return self.get_site_xpose('peg_tip')

    def get_object_xvel(self):
        return self.data.body('tool_1').cvel

    def get_target_xpose(self):
        return self.xtgt, np.eye(3)
        # return self.get_site_xpose('target_site')
        
    def get_coriolis_vector(self):
        # internal forces: Coriolis + gravitational
        return self.data.qfrc_bias[:self._nu]

    def set_robot_qpos(self, qpos_d):
        self.data.qpos[:self._nu] = qpos_d
    
    def get_robot_xvel(self):
        return self.get_jacobian_site(site=self.ee_site).dot(self.get_robot_qvel())
    
    def get_ft(self):
        return self.data.sensordata

    def get_joint_torques(self):
        return self.data.qfrc_actuator[:self._nu]

    def set_target_xpos(self, xpos=np.zeros(3), random_pose=True):
        # id = self.sim.model.body_name2id('repositioning_target')

        if np.linalg.norm(xpos) == 0:
            xtgt = np.array([0.65, 0.5, 0])
            if random_pose:
                sig = 1 if np.random.random() < 0.5 else -1
                xtgt[1] *= sig
                # xtgt += np.array([(np.random.rand()-0.5)/5, 
                #                   (np.random.rand()-0.5)/5,
                #                   0])
                xtgt += np.array([(np.random.rand()-0.5)/5, 
                                  (np.random.rand()-0.5)/5,
                                  0])
        else:
            xtgt = xpos
            xtgt[2] = 0
        
        self.xtgt = xtgt
        
        # jnt_adr = self.sim.model.body_jntadr[id]

        # # get the address of the joint angles in the data.qpos array
        # jnt_qpos_adr = self.sim.model.jnt_qposadr[jnt_adr]
        # self.sim.data.qpos[jnt_qpos_adr:jnt_qpos_adr + 3] = xtgt

        # # run mj_forward to propogate the change immediately
        # self.sim.forward()

    def set_object_xpos(self, xpos=np.zeros(3), random_pose=True):
        id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'repositioning_object')

        if np.linalg.norm(xpos) == 0:
            xobj = self.get_robot_xpose()[0] + np.array([0.1, 0, 0.75])
            if random_pose:
                xobj += np.array([(np.random.rand()-0.5)/10,
                                  (np.random.rand()-0.5)/10,
                                  (np.random.rand()-0.5)/10])
        else:
            xobj = xpos
        
        jnt_adr = self.model.body_jntadr[id]

        # get the address of the joint angles in the data.qpos array
        jnt_qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.data.qpos[jnt_qpos_adr:jnt_qpos_adr + 3] = xobj
        # set the new orientation
        self.data.qpos[jnt_qpos_adr + 3:jnt_qpos_adr + 7] = np.array([1, 0, 0, 0])

        # run mj_forward to propogate the change immediately
        self.forward()

    def reset_sim(self):
        mj.mj_resetData(self.model, self.data)

    def forward(self):
        mj.mj_forward(self.model, self.data)

    def render_frame(self, pos, mat):
        self.viewer.add_marker(pos=pos,
                          label='',
                          type=mj.generated.const.GEOM_SPHERE,
                          size=[.01, .01, .01])
        cylinder_half_height = 0.02
        pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
        self.viewer.add_marker(pos=pos_cylinder,
                          label='',
                          type=mj.generated.const.GEOM_CYLINDER,
                          size=[.005, .005, cylinder_half_height],
                          mat=mat)
        # self.viewer.render()
    
    def render_cartesian_frame(self, pos, mat, scale = 0.1, alpha = 1.):
        """ 
        Visualise a 3D coordinate frame.
        """
        self.viewer.add_marker(pos=pos,
                        label='',
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[.01, .01, .01])
        
        cylinder_half_height = scale
        pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
        self.viewer.add_marker(pos=pos_cylinder,
                        label='',
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[.005, .005, cylinder_half_height],
                        rgba=[0.,0.,1.,alpha],
                        mat=mat)
        
        pos_cylinder = pos + mat.dot([cylinder_half_height, 0.0, 0.])
        self.viewer.add_marker(pos=pos_cylinder,
                        label='',
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[cylinder_half_height, .005, .005 ],
                        rgba=[1., 0., 0., alpha],
                        mat=mat)
        
        pos_cylinder = pos + mat.dot([0.0, cylinder_half_height, 0.0])
        self.viewer.add_marker(pos=pos_cylinder,
                        label='',
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[.005, cylinder_half_height, .005],
                        rgba=[0., 1., 0., alpha],
                        mat=mat)
        
        self.viewer.add_marker(pos=self.xtgt,
                        label='',
                        type=mj.mjtGeom.mjGEOM_BOX,
                        size=[.15, .15, 0.005],
                        rgba=[1.,0.,0.,1],
                        mat=np.eye(3))
    
    def step_sim(self):
        mj.mj_step(self.model, self.data)
        if self.render:
            self.viewer.render()
            # # if mode == 'rgb_array':
            # mode = 'rgb_array'
            # self.viewer(mode).render(480, 640)
            # # window size used for old mujoco-py:
            # data = self.viewer(mode).read_pixels(480, 640, depth=False)
            # # original image is upside-down, so flip it
            # # return data[::-1, :, :]
            # print(data[::-1, :, :])
    
