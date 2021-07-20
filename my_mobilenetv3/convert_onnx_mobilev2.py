
import torch
import sys
sys.path.append('./model')
from MobileNetV2 import mobilenetv2



if __name__ == '__main__':
    onnx_model_path = './mobilenetv2.onnx'
    weight_path = r'/workspace/my_mobilenet_v3/output/MobileNetV2/MobileNetV2_best.pth'
    net = mobilenetv2(num_classes=3)
    net.load_state_dict(torch.load(weight_path))
    net.cuda()
    net.eval()
    dummy_input = torch.randn(1, 3, 224, 112).to("cuda")
    torch.onnx.export(net, (dummy_input), onnx_model_path, verbose=True, output_names=['_classifier'])



