import tensorflow as tf
from model import RDN
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TensorFlow RDN")
    parser.add_argument("--is_train", type=bool, default=True, help="if the train")
    parser.add_argument("--matlab_bicubic", type=bool, default=False, help="using bicubic interpolation in matlab")
    parser.add_argument("--image_size", type=int, default=32, help="the size of image input")
    parser.add_argument("--c_dim", type=int, default=3, help="the size of channel")
    parser.add_argument("--scale", type=int, default=3, help="the size of scale factor for preprocessing input image")
    parser.add_argument("--stride", type=int, default=16, help="the size of stride")
    parser.add_argument("--epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="the size of batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="the learning rate")
    parser.add_argument("--lr_decay_steps", type=int, default=10, help="steps of learning rate decay")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5, help="rate of learning rate decay")
    parser.add_argument("--is_eval", type=bool, default=True, help="if the evaluation")
    parser.add_argument("--test_img", type=str, default="", help="test_img")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="name of the checkpoint directory")
    parser.add_argument("--result_dir", type=str, default="result", help="name of the result directory")
    parser.add_argument("--train_set", type=str, default="DIV2K_train_HR", help="name of the train set")
    parser.add_argument("--test_set", type=str, default="Set5", help="name of the test set")
    parser.add_argument("--D", type=int, default=16, help="D")
    parser.add_argument("--C", type=int, default=8, help="C")
    parser.add_argument("--G", type=int, default=64, help="G")
    parser.add_argument("--G0", type=int, default=64, help="G0")
    parser.add_argument("--kernel_size", type=int, default=3, help="the size of kernel")
    return parser.parse_args()

def main():
    FLAGS = parse_args()
    
    rdn = RDN(is_train=FLAGS.is_train,
              is_eval=FLAGS.is_eval,
              image_size=FLAGS.image_size,
              c_dim=FLAGS.c_dim,
              scale=FLAGS.scale,
              batch_size=FLAGS.batch_size,
              D=FLAGS.D,
              C=FLAGS.C,
              G=FLAGS.G,
              G0=FLAGS.G0,
              kernel_size=FLAGS.kernel_size
              )

    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        if rdn.is_eval:
            rdn.eval(FLAGS)
        else:
            rdn.test(FLAGS)

if __name__ == '__main__':
    main()
