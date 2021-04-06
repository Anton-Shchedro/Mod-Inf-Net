import os
import shutil
import random
import argparse

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Dataset/ChineseAsPseudo/ChinaSet_AllFiles/Imgs',
                            help='Path to test data')
    parser.add_argument('--save_path', type=str, default='./Dataset/ChineseAsPseudo/ChinaSet_AllFiles/DataPrepare')
    opt = parser.parse_args()

    unlabeled_imgs_dir = '../../{}/'.format(opt.data_path)
    save_path = '../../{}/Imgs_splits'.format(opt.save_path)

    img_list = os.listdir(unlabeled_imgs_dir)
    random.shuffle(img_list)

    print(img_list)

    for i in range(0, int(len(img_list)/5)):
        choosed_img_lits = img_list[i*5:i*5+5]
        os.makedirs(save_path, exist_ok=True)
        for img_name in choosed_img_lits:
            save_split_path = os.path.join(save_path, 'image_{}'.format(i))
            os.makedirs(save_split_path, exist_ok=True)
            shutil.copyfile(os.path.join(unlabeled_imgs_dir, img_name),
                            os.path.join(save_split_path, img_name))
        print('[Processed] image_{}: {}'.format(i, choosed_img_lits))

if __name__ == "__main__":
    inference()