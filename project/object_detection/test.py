import onnx
model = onnx.load(r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\object_detection\yolov5\runs\train\exp3\weights\best.onnx')
print(onnx.helper.printable_graph(model.graph))
