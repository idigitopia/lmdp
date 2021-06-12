from .MDP_GPU import *
from .kernels import factored_vi_kernel_template

class FullMDPFactored(FullMDP):
    def __init__(self, A, build_args, solve_args):

        """
        :param A: Action Space of the MDP
        """
        super().__init__(A, build_args, solve_args)

        # Flag for factored. # ToDo remove this hack
        self.is_factored = True
        self.tmp_flag = True

        # MDP matrices CPU
        m_shape = (len(self.A), self.build_args.MAX_S_COUNT, self.build_args.MAX_NS_COUNT)
        self.costCountMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.costMatrix_cpu = np.zeros(m_shape).astype('float32')

        # MDP matrices GPU
        self.costMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))

        # Optimal Policy  parameters
        self.cD_cpu = np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32')  # Cost Vector / Penalty vector.
        self.cD_gpu = gpuarray.to_gpu(
            np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32'))  # value vector in gpu

        # Exploration Policy parameters
        self.e_cD_cpu = np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32')  # Cost Vector / Penalty vector.
        self.e_cD_gpu = gpuarray.to_gpu(
            np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32'))  # value vector in gpu

        # Safe Policy GPU parameters
        self.s_cD_cpu = np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32')  # Cost Vector / Penalty vector.
        self.s_cD_gpu = gpuarray.to_gpu(
            np.zeros((self.build_args.MAX_S_COUNT, 1)).astype('float32'))  # value vector in gpu

    def consume_transition(self, tran):
        """
        Adds the transition in the MDP
        """
        assert len(tran) == 5
        # pre-process transition
        s, a, ns, (r, c), d = tran
        ns = "end_state" if d else ns

        # Index states and get slot for transition
        self.index_if_new_state(s)
        self.index_if_new_state(ns)
        s_i, a_i, ns_i = self.s2i[s], self.a2i[a], self.s2i[ns]

        # Update MDP with new transition
        free_slots = np.where(self.tranidxMatrix_cpu[a_i, s_i] == 0)[0]
        if len(free_slots) >= 1:
            self.update_count_matrices(s_i, a_i, ns_i, r_sum=r, c_sum=c, count=1, slot=free_slots[0], append=True)
            self.update_prob_matrices(s_i, a_i)

    def update_count_matrices(self, s_i, a_i, ns_i, r_sum, c_sum, count, slot, append=False):
        if append:
            self.tranCountMatrix_cpu[a_i, s_i, slot] += count
            self.rewardCountMatrix_cpu[a_i, s_i, slot] += r_sum
            self.costCountMatrix_cpu[a_i, s_i, slot] += c_sum
        else:
            self.tranidxMatrix_cpu[a_i, s_i, slot] = ns_i
            self.tranCountMatrix_cpu[a_i, s_i, slot] = count
            self.rewardCountMatrix_cpu[a_i, s_i, slot] = r_sum
            self.costCountMatrix_cpu[a_i, s_i, slot] = c_sum

    def update_prob_matrices(self, s_i, a_i):
        # Normalize count Matrix
        self.tranProbMatrix_cpu[a_i, s_i] = self.tranCountMatrix_cpu[a_i, s_i] / (
                    np.sum(self.tranCountMatrix_cpu[a_i, s_i]) + 1e-12)
        self.rewardMatrix_cpu[a_i, s_i] = self.rewardCountMatrix_cpu[a_i, s_i] / (
                    self.tranCountMatrix_cpu[a_i, s_i] + 1e-12)
        self.costMatrix_cpu[a_i, s_i] = self.costCountMatrix_cpu[a_i, s_i] / (
                    self.tranCountMatrix_cpu[a_i, s_i] + 1e-12)

        self.e_rewardMatrix_cpu[a_i, s_i] = np.array(
            [self.get_rmax_reward_logic(s_i, a_i)] * self.build_args.MAX_NS_COUNT).astype("float32")
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

        self.tranProbMatrix_gpu = gpuarray.to_gpu(self.tranProbMatrix_cpu)
        self.tranidxMatrix_gpu = gpuarray.to_gpu(self.tranidxMatrix_cpu.astype("float32"))
        self.rewardMatrix_gpu = gpuarray.to_gpu(self.rewardMatrix_cpu)
        self.costMatrix_gpu = gpuarray.to_gpu(self.costMatrix_cpu)
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(self.e_rewardMatrix_cpu)
        
    def sync_opt_val_vectors_from_GPU(self):
        self.vD_cpu = self.vD_gpu.get()
        self.qD_cpu = self.qD_gpu.get()
        self.cD_cpu = self.cD_gpu.get()
        
    def sync_safe_val_vectors_from_GPU(self):
        self.s_vD_cpu = self.s_vD_gpu.get()
        self.s_qD_cpu = self.s_qD_gpu.get()
        self.s_cD_cpu = self.s_cD_gpu.get()
        
    def opt_bellman_backup_step_gpu(self):
        if self.tmp_flag:
            print("Updated backup operation called")
            self.tmp_flag = False

        # Temporary variables
        ACTION_COUNT, ROW_COUNT, COL_COUNT = self.tranProbMatrix_gpu.shape
        MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
        BLOCK_SIZE = 16

        # get the kernel code from the template
        kernel_code = factored_vi_kernel_template % {
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
            mod = compiler.SourceModule(kernel_code, cache_dir=os.getenv('PYCUDA_COMP_CACHE_DIR'))
        else:
            mod = compiler.SourceModule(kernel_code)
        matrixmul = mod.get_function("MatrixMulKernel")

        # Empty initialize Target Value and Q vectors
        tgt_vD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_cD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        
        matrixmul(
            # inputs
            self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.rewardMatrix_gpu, self.costMatrix_gpu,
            self.vD_gpu, self.cD_gpu,
            # output
            tgt_vD_gpu, tgt_cD_gpu, tgt_qD_gpu, tgt_error_gpu,
            grid=grid,
            # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
            block=(BLOCK_SIZE, BLOCK_SIZE, 1)
        )

        self.vD_gpu.gpudata.free()
        self.cD_gpu.gpudata.free()
        self.qD_gpu.gpudata.free()

        self.vD_gpu = tgt_vD_gpu
        self.cD_gpu = tgt_cD_gpu
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
        kernel_code = factored_vi_kernel_template % {
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
        tgt_cD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        matrixmul(
            # inputs
            self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.rewardMatrix_gpu, self.costMatrix_gpu,
            self.s_vD_gpu, self.s_cD_gpu,
            # output
            tgt_vD_gpu, tgt_cD_gpu, tgt_qD_gpu, tgt_error_gpu,
            grid=grid,
            # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
            block=(BLOCK_SIZE, BLOCK_SIZE, 1)
        )

        self.s_vD_gpu.gpudata.free()
        self.s_cD_gpu.gpudata.free()
        self.s_qD_gpu.gpudata.free()

        self.s_vD_gpu = tgt_vD_gpu
        self.s_cD_gpu = tgt_cD_gpu
        self.s_qD_gpu = tgt_qD_gpu


        self.s_gpu_backup_counter += 1
        if (self.s_gpu_backup_counter + 1) % 25 == 0:
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = max_error_gpu.get()
            max_error_gpu.gpudata.free()
            self.s_curr_vi_error = float(max_error)
        tgt_error_gpu.gpudata.free()

