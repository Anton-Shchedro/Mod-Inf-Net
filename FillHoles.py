import argparse
from Code.utils.dataloader_VisualTest import fill_holes


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Results/LungSegmentation/NewPseudo/MontFill',
                        help='Path to test data')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\n".format(opt), "#" * 20)

    gt_root = '{}/'.format(opt.data_path)
    #gt_root = '{}/GT/'.format(opt.data_path)

    print("start fill")

    fill_holes(gt_root)

    print("end fill")


if __name__ == "__main__":
    inference()
