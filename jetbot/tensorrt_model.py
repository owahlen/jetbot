import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from pycuda.tools import make_default_context


class TRTModel(object):

    def __init__(self, engine_path, final_shapes=None):
        """Initialize TensorRT engine and context."""
        cuda.init()
        self.cuda_context = make_default_context()

        # load engine
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_path)

        # allow override for reshaping each output
        self.final_shapes = final_shapes

        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, \
            self.bindings, self.input_names, self.output_names = self._allocate_buffers()
        except Exception as e:
            raise RuntimeError('failed to allocate CUDA resources') from e
        finally:
            self.cuda_context.pop()

    def __del__(self):
        """Free CUDA memory and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream
        self.runtime.__del__()
        self.context.__del__()
        self.engine.__del__()
        del self.logger

    def _load_engine(self, engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings, input_names, output_names = \
            [], [], [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            nptype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, nptype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                input_names.append(binding)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                output_names.append(binding)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings, input_names, output_names

    def execute(self, *inputs):
        """Detect objects in the input image(s)."""

        self.cuda_context.push()

        # copy inputs from host to device
        for i in range(len(self.host_inputs)):
            np.copyto(self.host_inputs[i], inputs[i].ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[i], self.host_inputs[i], self.stream)

        # execute inference
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)

        # copy outputs from cuda device back to host and reshape them as needed
        outputs = [None] * len(self.host_outputs)
        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
            if self.final_shapes is not None:
                shape = self.final_shapes[i]
            else:
                output_name = self.output_names[i]
                shape = self.engine.get_binding_shape(output_name)
            outputs[i] = self.host_outputs[i].reshape(shape)

        # synchronize async execution
        self.stream.synchronize()
        self.cuda_context.pop()
        return outputs

    def __call__(self, *inputs):
        return self.execute(*inputs)
