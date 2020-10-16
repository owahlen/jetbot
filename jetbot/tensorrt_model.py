import atexit

import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.autoinit  # this import initializes pycuda
import pycuda.driver as cuda
import tensorrt as trt


class TRTModel(object):

    def __init__(self, engine_path, final_shapes=None):
        # initialize runtime
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self._create_buffers()
        self.context = self.engine.create_execution_context()

        # allow override for reshaping each output
        self.final_shapes = final_shapes

        # destroy at exit
        atexit.register(self.destroy)

    def _create_buffers(self):
        self.host_inputs = []  # raveled data of each input as passed to the self.execute method
        self.cuda_inputs = []  # host_inputs copied to cuda device
        self.cuda_outputs = []  # outputs on cuda device
        self.host_outputs = []  # raveled cuda_outputs copied back to host
        self.input_names = []  # names of all input bindings
        self.output_names = []  # names of all output bindings
        self.bindings = [] # pointers to allocated cuda memory
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            nptype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, nptype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                self.input_names.append(binding)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                self.output_names.append(binding)
        self.outputs = [None] * len(self.host_outputs)

    def execute(self, *inputs):
        # copy inputs from host to device
        for i in range(len(self.host_inputs)):
            np.copyto(self.host_inputs[i], inputs[i].ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[i], self.host_inputs[i], self.stream)

        # execute inference
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)

        # copy outputs from cuda device back to host and reshape them as needed
        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
            if self.final_shapes is not None:
                shape = self.final_shapes[i]
            else:
                output_name = self.output_names[i]
                shape = self.engine.get_binding_shape(output_name)
            self.outputs[i] = self.host_outputs[i].reshape(shape)

        # synchronize async execution
        self.stream.synchronize()
        return self.outputs

    def __call__(self, *inputs):
        return self.execute(*inputs)

    def destroy(self):
        self.runtime.__del__()
        self.context.__del__()
        self.engine.__del__()
        del self.logger
