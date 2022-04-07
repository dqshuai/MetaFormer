from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
from config import get_inference_config
from models import build_model
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np
import argparse

try:
    from apex import amp
except ImportError:
    amp = None

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def model_config(config_path):
    args = Namespace(cfg=config_path)
    config = get_inference_config(args)
    return config


def read_class_names(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    class_list = []

    for l in lines:
        line = l.strip().split()
        # class_list.append(line[0])
        class_list.append(line[1][4:])

    classes = tuple(class_list)
    return classes


class GenerateEmbedding:
    def __init__(self, text_file):
        self.text_file = text_file

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def generate(self):
        text_list = []
        with open(self.text_file, 'r') as f_text:
            for line in f_text:
                line = line.encode(encoding='UTF-8', errors='strict')
                line = line.replace(b'\xef\xbf\xbd\xef\xbf\xbd', b' ')
                line = line.decode('UTF-8', 'strict')
                text_list.append(line)
            # data = f_text.read()
        select_index = np.random.randint(len(text_list))
        inputs = self.tokenizer(text_list[select_index], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=32)
        outputs = self.model(**inputs)
        embedding_mean = outputs[1].mean(dim=0).reshape(1, -1).detach().numpy()
        embedding_full = outputs[1].detach().numpy()
        embedding_words = outputs[0] # outputs[0].detach().numpy()
        return None, None, embedding_words


class Inference:
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.classes = ("cat", "dog")
        self.classes = read_class_names(r"D:\dataset\CUB_200_2011\CUB_200_2011\classes_custom.txt")

        self.config = model_config(self.config_path)
        self.model = build_model(self.config)
        self.checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(self.checkpoint['model'], strict=False)
        self.model.eval()
        self.model.cuda()

        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(), # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def infer(self, img_path, meta_data_path):
        _, _, meta = GenerateEmbedding(meta_data_path).generate()
        meta = meta.cuda()
        img = Image.open(img_path).convert('RGB')
        img = self.transform_img(img)
        img.unsqueeze_(0)
        img = img.cuda()
        img = Variable(img).to(self.device)
        out = self.model(img, meta)

        _, pred = torch.max(out.data, 1)
        predict = self.classes[pred.data.item()]
        # print(Fore.MAGENTA + f"The Prediction is: {predict}")
        return predict


def parse_option():
    parser = argparse.ArgumentParser('MetaFG Inference script', add_help=False)
    parser.add_argument('--cfg', type=str, default='D:/pycharmprojects/MetaFormer/configs/MetaFG_meta_bert_1_224.yaml', metavar="FILE", help='path to config file', )
    # easy config modification
    parser.add_argument('--model-path', default='D:\pycharmprojects\MetaFormer\output\MetaFG_meta_1\cub_200\ckpt_epoch_92.pth', type=str, help="path to model data")
    parser.add_argument('--img-path', default=r"D:\dataset\CUB_200_2011\CUB_200_2011\images\012.Yellow_headed_Blackbird\Yellow_Headed_Blackbird_0003_8337.jpg", type=str, help='path to image')
    parser.add_argument('--meta-path', default=r"D:\dataset\CUB_200_2011\text_c10\012.Yellow_headed_Blackbird\Yellow_Headed_Blackbird_0003_8337.txt", type=str, help='path to meta data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_option()
    result = Inference(config_path=args.cfg,
                       model_path=args.model_path).infer(img_path=args.img_path, meta_data_path=args.meta_path)
    print("Predicted: ", result)

# Usage: python inference.py --cfg 'path/to/cfg' --model_path 'path/to/model' --img-path 'path/to/img' --meta-path 'path/to/meta'