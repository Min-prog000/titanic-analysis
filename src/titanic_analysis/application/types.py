import numpy as np
import onnxruntime as ort

OutputItem = np.ndarray | ort.SparseTensor | list | dict | float | int
