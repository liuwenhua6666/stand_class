
import sys
sys.path.append('./model')
from model import MobileNetV3_large
from model import MobileNetV3_small
import torch

class Detector(object):
    # netkind为'large'或'small'可以选择加载MobileNetV3_large或MobileNetV3_small
    # 需要事先训练好对应网络的权重
    def __init__(self,net_kind,num_classes=17):
        super(Detector, self).__init__()
        kind=net_kind.lower()
        if kind=='large':
            self.net = MobileNetV3_large(num_classes=num_classes)
        elif kind=='small':
            self.net = MobileNetV3_large(num_classes=num_classes)
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def load_weights(self,weight_path):
        self.net.load_state_dict(torch.load(weight_path,map_location='cpu'))

if __name__ == '__main__':
    onnx_model_path = './mobilenetv3.onnx'
    weight_path = r'/workspace/my_mobilenet_v3/output/MobileNetV3_large/MobileNetV3_large_best.pth'
    detector=Detector('large',num_classes=3)
    detector.load_weights(weight_path=weight_path)
    detector.net.eval()
    dummy_input = torch.randn(1, 3, 224, 112).to("cuda")
    torch.onnx.export(detector.net, (dummy_input), onnx_model_path, verbose=True, output_names=['_classifier'])
