import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt
import time

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

CLASSES = ['person']

ObjThresh = 0.6
input_imgH = 640
input_imgW = 640
max_num = 100


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))


        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):

    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize the stream
    stream.synchronize()

    # Return only the host outputs.
    return [out.host for out in outputs]


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255.0

    image -= (0.485, 0.456, 0.406)
    image /= (0.229, 0.224, 0.225)

    image = image.transpose(2, 0, 1)
    img_input = image.copy()

    return img_input


def postprocess(pred_results, img_h, img_w):

    output = []
    for i in range(len(pred_results)):
        output.append(pred_results[i].reshape((-1)))

    pred_logits = output[0]
    pred_boxes = output[1]

    print('pred_logits:', len(pred_logits))
    print('pred_boxes:', len(pred_boxes))

    predBoxs = []
    for i in range(max_num):
        softmaxsum = 0
        for c in range(len(CLASSES) + 2):
            pred_logits[i * (len(CLASSES) + 2) + c] = exp(pred_logits[i * (len(CLASSES) + 2) + c])
            softmaxsum += pred_logits[i * (len(CLASSES) + 2) + c]

        for c in range(len(CLASSES) + 2):
            pred_logits[i * (len(CLASSES) + 2) + c] /= softmaxsum

        softmaxmax = 0
        softmaxindex = 0
        for c in range(len(CLASSES) + 2):
            if c == 0:
                softmaxmax = pred_logits[i * (len(CLASSES) + 2) + c]
                softmaxindex = c
            else:
                if softmaxmax < pred_logits[i * (len(CLASSES) + 2) + c]:
                    softmaxmax = pred_logits[i * (len(CLASSES) + 2) + c]
                    softmaxindex = c

        if softmaxmax > ObjThresh and softmaxindex == 1:
            x_c, y_c, w, h = pred_boxes[i * 4 + 0], pred_boxes[i * 4 + 1], pred_boxes[i * 4 + 2], pred_boxes[i * 4 + 3]
            box = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]

            rect = DetectBox(softmaxindex, softmaxmax, box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h)
            predBoxs.append(rect)

    return predBoxs


def main():
    engine_file_path = 'detr_r50_person_zq_sim_fp32.trt'
    image_path = 'test.jpg'

    orig = cv2.imread(image_path)
    img_h, img_w = orig.shape[:2]
    image = precess_image(orig, input_imgW, input_imgH)

    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        inputs[0].host = image
	
        t1 = time.time()
        for i in range(1000):
            trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        t2 = time.time()
        print('run fp32 model 1000 times average time :', (t2 - t1) / 1000.0)
        print('trt_outputs:', len(trt_outputs))

        out = []
        for i in range(len(trt_outputs)):
            out.append(trt_outputs[i])
        predbox = postprocess(out, img_h, img_w)

        print('obj num is :', len(predbox))

        for i in range(len(predbox)):
            xmin = int(predbox[i].xmin + 0.5)
            ymin = int(predbox[i].ymin + 0.5)
            xmax = int(predbox[i].xmax + 0.5)
            ymax = int(predbox[i].ymax + 0.5)
            classId = predbox[i].classId
            score = predbox[i].score

            cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            ptext = (xmin, ymin + 15)
            title = str(CLASSES[classId - 1]) + ":%.2f" % score
            cv2.putText(orig, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite('./test_result_tensorRT_fp32.jpg', orig)


if __name__ == '__main__':
    print('This is main ...')
    main()