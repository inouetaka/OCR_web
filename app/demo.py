import argparse
import time
import toml

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import CTCLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device ="cpu"


def option(root='./config/option.toml'):
    opt = toml.load(open(root))
    return opt['opt']


def character_loader():
    opt = option()
    with open(opt['character'], 'r')as ja:
        character = ja.read()

    return character

def loader():
    opt = option()

    """ model configuration """
    character = character_loader()
    converter = CTCLabelConverter(character)
    model = Model()
    model = torch.nn.DataParallel(model).to(device)

    # load model
    #print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt['saved_model'], map_location=device))

    length_for_pred = torch.IntTensor(opt['batch_max_length'] * opt['batch_size']).to(device)
    text_for_pred = torch.LongTensor(opt['batch_size'], opt['batch_max_length'] + 1).fill_(0).to(device)
    model.eval()

    return model, converter, length_for_pred, text_for_pred


# ------------------------------------------------------------------------------------------------------------------ #

def original_demo(model, converter, length_for_pred, text_for_pred):
    opt = option()
    AlignCollate_demo = AlignCollate(imgH=opt['imgH'], imgW=opt['imgW'], keep_ratio_with_pad=opt['PAD'])
    demo_data = RawDataset(root=opt['image_folder'], opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=int(opt['workers']),
        collate_fn=AlignCollate_demo, pin_memory=True)
    print(demo_loader)
    # predict


    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # 最大長予測用
            #torch.cuda.synchronize(device)
            if 'CTC' == opt['Prediction']:
                print('kotti')
                preds = model(image, text_for_pred).log_softmax(2)
                # 最大確率を選択し、インデックスを文字にデコードします
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # 最大確率を選択し、インデックスを文字にデコードします
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            print('-' * 80)
            print('image_path\tpredicted_labels')
            print('-' * 80)
            for img_name, pred in zip(image_path_list, preds_str):
                if 'Attn' == opt['Prediction']:
                    pred = pred[:pred.find('[s]')]  # 文の終わりトークン（[s]）の後の剪定

                print(f'{img_name}\t{pred}')





if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = True
    num_gpu = torch.cuda.device_count()

    model, converter, length_for_pred, text_for_pred = loader()
    for i in range(5):
        original_demo(model, converter, length_for_pred, text_for_pred)