from collections import defaultdict
import time
import math as mth
from .kernels import vi_kernel_template
import numpy as np
from copy import deepcopy as cpy
from collections import Counter
import os 

from pycuda import compiler, gpuarray
import pycuda.autoinit

def init2dict():
    return {}

def init2list():
    return []


def init2zero():
    return 0


def init2zero_def_dict():
    return defaultdict(init2zero)


def init2zero_def_def_dict():
    return defaultdict(init2zero_def_dict)


class FullMDP(object):
    def __init__(self, A, build_args, solve_args):

        """
        :param A: Action Space of the MDP
        """

        # Flag for factored.
        self.is_factored = False

        has_attributes = lambda v, a_list: all([hasattr(v, a) for a in a_list])
        assert has_attributes(build_args, ["ur", "MAX_S_COUNT", "MAX_NS_COUNT"])
        assert has_attributes(solve_args, ["gamma", "slip_prob", "default_mode"])

        self.omit_list = ["end_state", "unknown_state"]
        self.build_args = build_args
        self.solve_args = solve_args

        # VI CPU/GPU parameters
        self.curr_vi_error = float("inf")  # Backup error
        self.e_curr_vi_error = float("inf")  # exploration Backup error
        self.s_curr_vi_error = float("inf")  # safe  Backup error

        # MDP Parameters
        self.A = A

        # MDP dict to matrix api Parameters # i = index
        self.s2i = {"unknown_state": 0, "end_state": 1}
        self.i2s = {0: "unknown_state", 1: "end_state"}
        self.a2i = {a: i for i, a in enumerate(self.A)}
        self.i2a = {i: a for i, a in enumerate(self.A)}
        self.free_i = list(reversed(range(2, self.build_args.MAX_S_COUNT)))
        self.filled_mask = np.zeros((self.build_args.MAX_S_COUNT,)).astype('uint')
        self.filled_mask[[0, 1]] = 1  # set filled for unknown and end state

        # MDP matrices CPU
        m_shape = (len(self.A), self.build_args.MAX_S_COUNT, self.build_args.MAX_NS_COUNT)
        self.tranCountMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.tranidxMatrix_cpu = np.zeros(m_shape).astype('uint')
        self.tranProbMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.rewardCountMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.rewardMatrix_cpu = np.zeros(m_shape).astype('float32')

        # MDP matrices GPU
        self.tranProbMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))
        self.tranidxMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))
        self.rewardMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))

        # Initialize for unknown and end states
        self._initialize_end_and_unknown_state()

        # Help :
        # self.tranCountMatrix_cpu = count Matrix [a_idx, s_idx, ns_id_idx] [Holds tran Counts] ,
        # self.tranidxMatrix_cpu = id Matrix [a_idx, s_idx, ns_id_idx] [Holds ns_idx]
        # self.tranProbMatrix_cpu =  prob Matrix [a_idx, s_idx, ns_id_idx] [Holds tran probabilities]
        # self.rewardCountMatrix_cpu = count Matrix [a_idx, s_idx, ns_id_idx] [Holds reward Counts]
        # self.rewardMatrix_cpu = reward Matrix [a_idx, s_idx, ns_id_idx] [Holds normalized rewards]

        # Optimal Policy  parameters
        self.pD_cpu = np.zeros((self.build_args.MAX_S_COUNT,)).astype('uint')  # Optimal Policy Vector
        self.vD_cpu = np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32')
        self.qD_cpu = np.zeros((self.build_args.MAX_S_COUNT, len(self.A))).astype('float32')
        self.vD_gpu = gpuarray.to_gpu(np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32'))  # value vector in gpu
        self.qD_gpu = gpuarray.to_gpu(np.zeros((self.build_args.MAX_S_COUNT, len(self.A))).astype('float32'))  # q matrix in gpu
        self.gpu_backup_counter = 0

        # Exploration Policy parameters
        self.e_pD_cpu = np.zeros((self.build_args.MAX_S_COUNT,)).astype('uint')  # Optimal Policy Vector
        self.e_vD_gpu = gpuarray.to_gpu(np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32'))
        self.e_qD_gpu = gpuarray.to_gpu(np.zeros((self.build_args.MAX_S_COUNT, len(self.A))).astype('float32'))
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))
        self.e_vD_cpu = np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32')
        self.e_qD_cpu = np.zeros((self.build_args.MAX_S_COUNT, len(self.A))).astype('float32')
        self.e_rewardMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.e_gpu_backup_counter = 0

        # Safe Policy GPU parameters
        self.s_pD_cpu = np.zeros((self.build_args.MAX_S_COUNT,)).astype('uint')  # Optimal Policy Vector
        self.s_vD_gpu = gpuarray.to_gpu(np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32'))
        self.s_qD_gpu = gpuarray.to_gpu(np.zeros((self.build_args.MAX_S_COUNT, len(self.A))).astype('float32'))
        self.s_vD_cpu = np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32')
        self.s_qD_cpu = np.zeros((self.build_args.MAX_S_COUNT, len(self.A))).astype('float32')
        self.s_gpu_backup_counter = 0

        # cached items
        # self.refresh_cache_dicts()

    def _initialize_end_and_unknown_state(self):
        self.tranidxMatrix_cpu[:, :, 0] = 0  # [a_idx, s_idx, ns_id_idx] # everything goes to unknown state
        self.tranCountMatrix_cpu[:, :, 0] = 1  # [a_idx, s_idx, ns_id_idx] # everything goes to unknown state
        self.tranProbMatrix_cpu[:, :, 0] = 1  # [a_idx, s_idx, ns_id_idx] # everything goes to unknown state
        self.rewardCountMatrix_cpu[:, :, 0] = self.build_args.ur  # [a_idx, s_idx, ns_id_idx] # everything  has ur rewards
        self.rewardMatrix_cpu[:, :, 0] = self.build_args.ur  # [a_idx, s_idx, ns_id_idx] # everything  has ur rewards

        self.tranidxMatrix_cpu[:, 0, 0] = 0  # unknown state has a self loop
        self.tranCountMatrix_cpu[:, 0, 0] = 1  # unknown state has a self loop
        self.tranProbMatrix_cpu[:, 0, 0] = 1  # unknown state has a self loop
        self.rewardCountMatrix_cpu[:, 0, 0] = 0  # unknown state self loop has no rewards
        self.rewardMatrix_cpu[:, 0, 0] = 0  # unknown state self loop has no rewards

        self.tranidxMatrix_cpu[:, 1, 0] = 1  # end state has a self loop
        self.tranCountMatrix_cpu[:, 1, 0] = 1  # end state has a self loop
        self.tranProbMatrix_cpu[:, 1, 0] = 1  # end state has a self loop
        self.rewardCountMatrix_cpu[:, 1, 0] = 0  # end state self loop has no rewards
        self.rewardMatrix_cpu[:, 1, 0] = 0  # end state has self loop no rewards


    def get_free_index(self):
        """Used to assign index to a new state"""
        indx = self.free_i.pop()
        self.filled_mask[indx] = 1
        return indx

    def index_if_new_state(self, s):
        is_new = s not in self.s2i
        if is_new:
            i = self.get_free_index()
            self.s2i[s], self.i2s[i] = i, s
        return is_new

    def consume_transition(self, tran):
        """
        Adds the transition in the MDP
        """
        assert len(tran) == 5
        # pre-process transition
        s, a, ns, r, d = tran
        ns = "end_state" if d else ns

        # Index states and get slot for transition
        self.index_if_new_state(s)
        self.index_if_new_state(ns)
        s_i, a_i, ns_i = self.s2i[s], self.a2i[a], self.s2i[ns]

        # Update MDP with new transition
        # Have we seen this state action before? 
        slot_idx = self.get_action_slot(s_i, a_i, ns_i)
        self.update_count_matrices(s_i, a_i, ns_i, r_sum=r, count=1, slot=slot_idx)
        self.update_prob_matrices(s_i, a_i)

    def get_action_slot(self, s_i, a_i, ns_i):
        """[summary]
        For a given state and action, returns a slot for next_state. 
        Args:
            s_i ([type]): [description]
            a_i ([type]): [description]
            ns_i ([type]): [description]
        """

        slot_already_assigned = ns_i in self.tranidxMatrix_cpu[a_i, s_i]

        if slot_already_assigned:
            occupied_slot_idx = np.where(self.tranidxMatrix_cpu[a_i, s_i] == ns_i)[0][0]
            ret_slot = occupied_slot_idx
        else:
            free_slots = np.where(self.tranidxMatrix_cpu[a_i, s_i] == 0)[0]
            next_free_slot_idx = free_slots[0]
            ret_slot = next_free_slot_idx

        # if s_i, a_i has been already seen return the slot allready assigned. 
        return ret_slot

    def update_count_matrices(self, s_i, a_i, ns_i, r_sum, c_sum, count, slot, override = False):
        
        slot_empty = self.tranCountMatrix_cpu[a_i, s_i, slot] == 0

        if override or slot_empty:
            self.tranidxMatrix_cpu[a_i, s_i, slot] = ns_i
            self.tranCountMatrix_cpu[a_i, s_i, slot] = count
            self.rewardCountMatrix_cpu[a_i, s_i, slot] = r_sum
            self.costCountMatrix_cpu[a_i, s_i, slot] = c_sum

        else:
            can_be_appended = self.tranidxMatrix_cpu[a_i, s_i, slot] == ns_i
            assert can_be_appended, "Someting is wrong here, slot already occupied by something else"
            self.tranCountMatrix_cpu[a_i, s_i, slot] += count
            self.rewardCountMatrix_cpu[a_i, s_i, slot] += r_sum
            self.costCountMatrix_cpu[a_i, s_i, slot] += c_sum

    def update_prob_matrices(self, s_i, a_i):
        # Normalize count Matrix
        self.tranProbMatrix_cpu[a_i, s_i] = self.tranCountMatrix_cpu[a_i, s_i] / (np.sum(self.tranCountMatrix_cpu[a_i, s_i]) + 1e-12)
        self.rewardMatrix_cpu[a_i, s_i] = self.rewardCountMatrix_cpu[a_i, s_i] / (self.tranCountMatrix_cpu[a_i, s_i] + 1e-12)
        self.e_rewardMatrix_cpu[a_i, s_i] = np.array([self.get_rmax_reward_logic(s_i, a_i)] * self.build_args.MAX_NS_COUNT).astype("float32")
        # assert len(self.tranProbMatrix_cpu[i][j]) == len(self.tranidxMatrix_cpu[i][j])

    def sync_mdp_from_cpu_to_gpu(self, ):
        # self.tranCountMatrix_gpu.gpudata.free()
        try:
            self.tranProbMatrix_gpu.gpudata.free()
            self.tranidxMatrix_gpu.gpudata.free()
            self.rewardMatrix_gpu.gpudata.free()
            self.e_rewardMatrix_gpu.gpudata.free()
        except:
            print("free failed")

        # self.tranCountMatrix_gpu = gpuarray.to_gpu(self.tranCountMatrix_cpu)
        self.tranProbMatrix_gpu = gpuarray.to_gpu(self.tranProbMatrix_cpu)
        self.tranidxMatrix_gpu = gpuarray.to_gpu(self.tranidxMatrix_cpu.astype("float32"))
        self.rewardMatrix_gpu = gpuarray.to_gpu(self.rewardMatrix_cpu)
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(self.e_rewardMatrix_cpu)

    def get_rmax_reward_logic(self, s, a):
        # get the sum of distances for k nearest neighbor
        # pick the top 10%
        # set all unknwon sa for this s as rmax  actions as rmax.
        return 0  # todo add some logic here

    def sync_opt_val_vectors_from_GPU(self):
        tmp_vD_cpu = self.vD_gpu.get()
        tmp_qD_cpu = self.qD_gpu.get()
        self.vD_cpu = tmp_vD_cpu
        self.qD_cpu = tmp_qD_cpu

    def sync_explr_val_vectors_from_GPU(self):
        tmp_e_vD_cpu = self.e_vD_gpu.get()
        tmp_e_qD_cpu = self.e_qD_gpu.get()
        self.e_vD_cpu = tmp_e_vD_cpu
        self.e_qD_cpu = tmp_e_qD_cpu

    def sync_safe_val_vectors_from_GPU(self):
        tmp_s_vD_cpu = self.s_vD_gpu.get()
        tmp_s_qD_cpu = self.s_qD_gpu.get()
        self.s_vD_cpu = tmp_s_vD_cpu
        self.s_qD_cpu = tmp_s_qD_cpu

    def do_backup(self, mode, module, n_backups):
        bkp_fxn_dict = {"optimal": {"CPU": self.opt_bellman_backup_step_cpu, "GPU": self.opt_bellman_backup_step_gpu},
                        "safe": {"CPU": self.safe_bellman_backup_step_cpu, "GPU": self.safe_bellman_backup_step_gpu},
                        "exploration": {"CPU": self.explr_bellman_backup_step_cpu,
                                        "GPU": self.explr_bellman_backup_step_gpu}}
        sync_fxn_dict = {"optimal": {"GPU": self.sync_opt_val_vectors_from_GPU},
                         "safe": {"GPU": self.sync_safe_val_vectors_from_GPU},
                         "exploration": {"GPU": self.sync_explr_val_vectors_from_GPU}}

        if mode == "CPU":
            for _ in range(n_backups):
                bkp_fxn_dict[module]["CPU"]()
        elif mode == "GPU":
            self.sync_mdp_from_cpu_to_gpu()
            for _ in range(n_backups):
                bkp_fxn_dict[module]["GPU"]()
            sync_fxn_dict[module]["GPU"]()
        else:
            print("Illegal Mode: Not Specified")
            assert False

    def do_optimal_backup(self, mode="CPU", n_backups=1):
        self.do_backup(mode=mode, module="optimal", n_backups=n_backups)

    def do_safe_backup(self, mode="CPU", n_backups=1):
        self.do_backup(mode=mode, module="safe", n_backups=n_backups)

    def do_explr_backup(self, mode="CPU", n_backups=1):
        self.do_backup(mode=mode, module="exploration", n_backups=n_backups)

    def get_state_count(self):
        return len(self.s2i)

    def solve(self, eps=1e-5, mode=None, safe_bkp=False, explr_bkp=False, verbose=True, reset_error = False):

        if reset_error:
            self.curr_vi_error = 99999
            
        mode = mode or self.solve_args["default_mode"]

        st = time.time()
        curr_error = self.curr_vi_error
        while abs(self.curr_vi_error) > eps:
            self.do_optimal_backup(mode=mode, n_backups=250)
            if safe_bkp:
                self.do_safe_backup(mode=mode, n_backups=250)
            if explr_bkp:
                self.do_explr_backup(mode=mode, n_backups=250)

            if self.curr_vi_error < curr_error / 10 and verbose:
                print("Elapsed Time:{}s, VI Error:{}, #Backups: {}".format(int(time.time() - st),
                                                                           round(self.curr_vi_error, 8),
                                                                           self.gpu_backup_counter))
                curr_error = self.curr_vi_error
        et = time.time()
        if verbose: print("Time takedn to solve", et - st)


    def __len__(self):
        return np.sum(self.filled_mask)

    def opt_bellman_backup_step_cpu(self):
        backup_error = 0
        for s, s_i in self.s2i.items():
            for a, a_i in self.a2i.items():
                ns_values = np.array([self.vD_cpu[ns_i] for ns_i in self.tranidxMatrix_cpu[a_i, s_i]]).squeeze()
                expected_ns_val = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * ns_values)
                expected_reward = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * self.rewardMatrix_cpu[a_i, s_i])
                self.qD_cpu[s_i, a_i] = expected_reward + self.solve_args.gamma * expected_ns_val

            new_val = np.max(self.qD_cpu[s_i])

            backup_error = max(backup_error, abs(new_val - self.vD_cpu[s_i]))
            self.vD_cpu[s_i] = new_val
            self.pD_cpu[s_i] = np.argmax(self.qD_cpu[s_i])

        self.curr_vi_error = backup_error
        return backup_error

    def safe_bellman_backup_step_cpu(self):
        backup_error = 0
        for s, s_i in self.s2i.items():
            for a, a_i in self.a2i.items():
                ns_values = np.array([self.s_vD_cpu[ns_i] for ns_i in self.tranidxMatrix_cpu[a_i, s_i]]).squeeze()
                expected_ns_val = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * ns_values)
                expected_reward = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * self.rewardMatrix_cpu[a_i, s_i])
                self.s_qD_cpu[s_i, a_i] = expected_reward + self.solve_args.gamma * expected_ns_val

            max_q, sum_q = np.max(self.s_qD_cpu[s_i]), np.sum(self.s_qD_cpu[s_i])
            new_val = (1 - self.solve_args.slip_prob) * max_q + self.solve_args.slip_prob * (sum_q - max_q)

            backup_error = max(backup_error, abs(new_val - self.s_vD_cpu[s_i]))
            self.s_vD_cpu[s_i] = new_val
            self.s_pD_cpu[s_i] = np.argmax(self.s_qD_cpu[s_i])

        self.s_curr_vi_error = backup_error

    def explr_bellman_backup_step_cpu(self):
        backup_error = 0
        for s, s_i in self.s2i.items():
            for a, a_i in self.a2i.items():
                ns_values = np.array([self.e_vD_cpu[ns_i] for ns_i in self.tranidxMatrix_cpu[a_i, s_i]]).squeeze()
                expected_ns_val = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * ns_values)
                expected_reward = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * self.e_rewardMatrix_cpu[a_i, s_i])
                self.e_qD_cpu[s][a] = expected_reward + self.solve_args.gamma * expected_ns_val

            new_val = np.max(self.qD_cpu[s_i])

            backup_error = max(backup_error, abs(new_val - self.e_vD_cpu[s]))
            self.e_vD_cpu[s_i] = new_val
            self.e_pD_cpu[s_i] = max(self.e_qD_cpu[s], key=self.e_qD_cpu[s].get)

        self.e_curr_vi_error = backup_error

    def opt_bellman_backup_step_gpu(self):
        # Temporary variables
        ACTION_COUNT, ROW_COUNT, COL_COUNT = self.tranProbMatrix_gpu.shape
        MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
        BLOCK_SIZE = 16

        # get the kernel code from the template
        kernel_code = vi_kernel_template % {
            'ROW_COUNT': ROW_COUNT,
            'COL_COUNT': COL_COUNT,
            'ACTION_COUNT': ACTION_COUNT,
            'MATRIX_SIZE': MATRIX_SIZE,
            'GAMMA': self.solve_args.gamma,
            'SLIP_ACTION_PROB': 0,
        }

        # Get grid dynamically by specifying the constant MATRIX_SIZE
        if MATRIX_SIZE % BLOCK_SIZE != 0:
            grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
        else:
            grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

        # compile the kernel code and get the compiled module
        if 'PYCUDA_COMP_CACHE_DIR' in os.environ:
            mod = compiler.SourceModule(kernel_code, cache_dir = os.getenv('PYCUDA_COMP_CACHE_DIR'))
        else:
            mod = compiler.SourceModule(kernel_code)
        matrixmul = mod.get_function("MatrixMulKernel")

        # Empty initialize Target Value and Q vectors
        tgt_vD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        try:
            matrixmul(
                # inputs
                self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.rewardMatrix_gpu, self.vD_gpu,
                # output
                tgt_vD_gpu, tgt_qD_gpu, tgt_error_gpu,
                grid=grid,
                # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
                block=(BLOCK_SIZE, BLOCK_SIZE, 1)
            )
        except:
            if (input("d for debugging") == 'd'):
                print(BLOCK_SIZE, BLOCK_SIZE, 1)
                import pdb;
                pdb.set_trace()

        self.vD_gpu.gpudata.free()
        self.qD_gpu.gpudata.free()
        self.vD_gpu = tgt_vD_gpu
        self.qD_gpu = tgt_qD_gpu

        self.gpu_backup_counter += 1
        if (self.gpu_backup_counter + 1) % 25 == 0:
            # print("checkingggg for epsilng stop")
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = max_error_gpu.get()
            max_error_gpu.gpudata.free()
            self.curr_vi_error = float(max_error)
        tgt_error_gpu.gpudata.free()

    def safe_bellman_backup_step_gpu(self):
        # print("Old backup operation called")

        # Temporary variables
        ACTION_COUNT, ROW_COUNT, COL_COUNT = self.tranProbMatrix_gpu.shape
        MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
        BLOCK_SIZE = 16

        # get the kernel code from the template
        kernel_code = vi_kernel_template % {
            'ROW_COUNT': ROW_COUNT,
            'COL_COUNT': COL_COUNT,
            'ACTION_COUNT': ACTION_COUNT,
            'MATRIX_SIZE': MATRIX_SIZE,
            'GAMMA': self.solve_args.gamma,
            'SLIP_ACTION_PROB': self.solve_args.slip_prob,
        }

        # Get grid dynamically by specifying the constant MATRIX_SIZE
        if MATRIX_SIZE % BLOCK_SIZE != 0:
            grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
        else:
            grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

        # compile the kernel code and get the compiled module
        if 'PYCUDA_COMP_CACHE_DIR' in os.environ:
            mod = compiler.SourceModule(kernel_code, cache_dir = os.getenv('PYCUDA_COMP_CACHE_DIR'))
        else:
            mod = compiler.SourceModule(kernel_code)
        matrixmul = mod.get_function("MatrixMulKernel")

        # Empty initialize Target Value and Q vectors
        tgt_vD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        try:
            matrixmul(
                # inputs
                self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.rewardMatrix_gpu, self.s_vD_gpu,
                # output
                tgt_vD_gpu, tgt_qD_gpu, tgt_error_gpu,
                grid=grid,
                # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
                block=(BLOCK_SIZE, BLOCK_SIZE, 1)
            )
        except:
            if (input("d for debugging") == 'd'):
                print(BLOCK_SIZE, BLOCK_SIZE, 1)
                import pdb;
                pdb.set_trace()

        self.s_vD_gpu.gpudata.free()
        self.s_qD_gpu.gpudata.free()
        self.s_vD_gpu = tgt_vD_gpu
        self.s_qD_gpu = tgt_qD_gpu

        self.s_gpu_backup_counter += 1
        if (self.s_gpu_backup_counter + 1) % 25 == 0:
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = max_error_gpu.get()
            max_error_gpu.gpudata.free()
            self.s_curr_vi_error = float(max_error)
        tgt_error_gpu.gpudata.free()

    def explr_bellman_backup_step_gpu(self):
        # Temporary variables
        ACTION_COUNT, ROW_COUNT, COL_COUNT = self.tranProbMatrix_gpu.shape
        MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
        BLOCK_SIZE = 16

        # get the kernel code from the template
        kernel_code = vi_kernel_template % {
            'ROW_COUNT': ROW_COUNT,
            'COL_COUNT': COL_COUNT,
            'ACTION_COUNT': ACTION_COUNT,
            'MATRIX_SIZE': MATRIX_SIZE,
            'GAMMA': self.solve_args.gamma,
            'SLIP_ACTION_PROB': 0,
        }

        # Get grid dynamically by specifying the constant MATRIX_SIZE
        if MATRIX_SIZE % BLOCK_SIZE != 0:
            grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
        else:
            grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

        # compile the kernel code and get the compiled module
        if 'PYCUDA_COMP_CACHE_DIR' in os.environ:
            mod = compiler.SourceModule(kernel_code, cache_dir = os.getenv('PYCUDA_COMP_CACHE_DIR'))
        else:
            mod = compiler.SourceModule(kernel_code)
        matrixmul = mod.get_function("MatrixMulKernel")

        # Empty initialize Target Value and Q vectors
        tgt_vD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        try:
            matrixmul(
                # inputs
                self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.e_rewardMatrix_gpu, self.e_vD_gpu,
                # output
                tgt_vD_gpu, tgt_qD_gpu, tgt_error_gpu,
                grid=grid,
                # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
                block=(BLOCK_SIZE, BLOCK_SIZE, 1)
            )
        except:
            if (input("d for debugging") == 'd'):
                print(BLOCK_SIZE, BLOCK_SIZE, 1)
                import pdb;
                pdb.set_trace()

        self.e_vD_gpu.gpudata.free()
        self.e_qD_gpu.gpudata.free()
        self.e_vD_gpu = tgt_vD_gpu
        self.e_qD_gpu = tgt_qD_gpu

        self.e_gpu_backup_counter += 1
        if (self.e_gpu_backup_counter + 1) % 25 == 0:
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = float(max_error_gpu.get())
            max_error_gpu.gpudata.free()
            self.e_curr_vi_error = max_error
        tgt_error_gpu.gpudata.free()

    # Debugging / Utilities
    def refresh_cache_dicts(self):
        self.tD = defaultdict(init2zero_def_def_dict)  # Transition Probabilities
        self.rD = defaultdict(init2zero_def_def_dict)  # Transition Probabilities

        for s, s_i in self.s2i.items():
            for a in self.A:
                for ns_slot, ns_i in enumerate(self.tranidxMatrix_cpu[self.a2i[a], s_i]):
                    ns, a_i = self.i2s[ns_i], self.a2i[a]
                    if self.tranProbMatrix_cpu[a_i, s_i, ns_slot] > 0:
                        # print(f"ns:{ns},ns_i:{ns_i}, a_i:{a_i}, a:{a}, s:{s}, s_i:{s_i} ,  ns_slot:{ns_slot}")
                        self.tD[s][a][ns] = self.tranProbMatrix_cpu[a_i, s_i, ns_slot]
                        self.rD[s][a][ns] = self.rewardMatrix_cpu[a_i, s_i, ns_slot]

    @property
    def missing_state_action_count(self):
        return sum([1 for s in self.rD for a in self.rD[s] if self.build_args.ur == self.rD[s][a]])

    @property
    def total_state_action_count(self):
        return len(self.tD) * len(self.A)

    @property
    def missing_state_action_percentage(self):
        return round(self.missing_state_action_count / self.total_state_action_count, 4)

    @property
    def valueDict(self):
        return {s: float(self.vD_cpu[i]) for s, i in self.s2i.items()}

    @property
    def s_valueDict(self):
        return {s: float(self.s_vD_cpu[i]) for s, i in self.s2i.items()}

    @property
    def qvalDict(self):
        return {s: {a: self.qD_cpu[i][j] for a, j in self.a2i.items()} for s, i in self.s2i.items()}

    @property
    def s_qvalDict(self):
        return {s: {a: self.s_qD_cpu[i][j] for a, j in self.a2i.items()} for s, i in self.s2i.items()}

    @property
    def polDict(self):
        qvalDict = self.qvalDict
        return {s: max(qvalDict[s], key = qvalDict[s].get) for s, i in self.s2i.items()}

    @property
    def s_polDict(self):
        qvalDict = self.s_qvalDict
        return {s: max(qvalDict[s], key = qvalDict[s].get) for s, i in self.s2i.items()}

    @property
    def qval_distribution(self):
        qvalDict = cpy(self.qvalDict)
        return [qsa for qs in qvalDict for qsa in qs]

    @property
    def unknown_state_count(self):
        return sum([self.check_if_unknown(s) for s in self.tD])

    @property
    def unknown_state_action_count(self):
        return sum([self.check_if_unknown(s, a) for s in self.tD for a in self.tD[s]])

    @property
    def fully_unknown_state_count(self):
        return sum([all([self.check_if_unknown(s, a) for a in self.tD[s]]) for s in self.tD])

    @property
    def fully_unknown_states(self):
        return [s for s in self.tD if all([self.check_if_unknown(s, a) for a in self.tD[s]])]

    def check_if_unknown(self, s, a=None):
        if s == "unknown_state":
            return False
        if a is not None:
            return "unknown_state" in self.tD[s][a]
        else:
            return sum([1 for a in self.A if "unknown_state" in self.tD[s][a]]) == len(self.A)

    def get_seen_action_count(self, s):
        return sum([1 for a in self.tC[s] if "unknown_state" not in self.tC[s][a]])

    def get_explored_action_count(self, s):
        return sum([1 for a in self.tC[s] if sum(self.tC[s][a].values()) > self.build_args.rmax_thres])

    @property
    def tran_prob_distribution(self):
        return [self.tD[s][a][ns] for s in self.tD for a in self.tD[s] for ns in self.tD[s][a]]

    @property
    def reward_distribution(self):
        return [self.rD[s][a] for s in self.rD for a in self.rD[s]]

    @property
    def state_action_count_distribution(self):
        return [sum(self.tC[s][a].values()) for s in self.tC for a in self.tC[s]]

    @property
    def state_action_fan_out_distribution(self):
        return [len(self.tD[s][a]) for s in self.tD for a in self.tD[s]]

    @property
    def state_action_fan_in_distribution(self):
        list_of_ns = [ns for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if ns not in self.omit_list]
        counter = Counter(list_of_ns)
        return list(counter.values())

    @property
    def explored_action_count_distribution(self):
        return [sum([1 for a in self.tC[s] if sum(self.tC[s][a].values()) > self.build_args.rmax_thres]) for s in
                self.tC]

    @property
    def seen_action_count_distribution(self):
        return [sum([1 for a in self.tC[s] if "unknown_state" not in self.tC[s][a]]) for s in self.tC]

    @property
    def state_count_distribution(self):
        return [sum([sum(self.tC[s][a].values()) for a in self.tC[s]]) for s in self.tC]

    @property
    def self_loop_prob_distribution(self):
        return [self.tD[s][a][ns] for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if s == ns]

    @property
    def self_loop_count(self):
        return sum([1 for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if s == ns])

    @property
    def end_state_state_action_count(self):
        return sum([1 for s in self.tD for a in self.tD[s] if "end_state" in self.tD[s][a]])

    def get_state_info(self,s):
        return 1
        
