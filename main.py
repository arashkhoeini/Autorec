from data_preprocessor import *
from AutoRec import AutoRec
import tensorflow as tf
import time
import argparse
import math
current_time = time.time()

parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int,default=100)

parser.add_argument('--optimizer_method', choices=['Adam','RMSProp'],default='Adam')
parser.add_argument('--grad_clip', type=bool,default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50,help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--data_set', type=str, default='100k') #another option is '1m'
parser.add_argument('--upl', type=int, default=10)
parser.add_argument('--repeat_number' , type=int, default=5)

args = parser.parse_args()
tf.set_random_seed(args.random_seed)
#np.random.seed(args.random_seed)

data_name = 'ml-100k';

if args.data_set == '100k':
    data_name = 'ml-100k';
    num_users = 943
    num_items = 1682

elif args.data_set == '1m':
    data_name = 'ml-1m'
    num_users = 6040
    num_items = 3952


path = "./data/%s" % data_name + "/"

ndcg5_list = []
ndcg10_list = []
for _ in range(args.repeat_number):
    tf.reset_default_graph()
    R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
    user_train_set,item_train_set,user_test_set,item_test_set \
        = read_rating(path, num_users,num_items, 1, 0, args.upl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        autoRec = AutoRec(sess,args,
                          num_users,num_items,
                          R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                          user_train_set, item_train_set, user_test_set, item_test_set)
        autoRec.run()
        ndcg5_list.append(autoRec.test_model(n = 5))
        ndcg10_list.append(autoRec.test_model(n = 10))
#sess.close()


ndcg5_avg = (sum(ndcg5_list)/args.repeat_number)
ndcg10_avg = (sum(ndcg10_list) / args.repeat_number)
temp = 0
for ndcg in ndcg5_list:
    temp += math.pow(ndcg - ndcg5_avg , 2)
std5 = math.sqrt(temp/args.repeat_number)
temp = 0
for ndcg in ndcg10_list:
    temp += math.pow(ndcg - ndcg10_avg, 2)
std10 = math.sqrt(temp / args.repeat_number)

print("UPl: %s "%args.upl)

print("List of NDCG@5: %s "%ndcg5_list)
print("Average in NDCG@5 is %s"%ndcg5_avg)
print("Standard Deviation is %s"%std5)

print('List of NDCG@10: %s'%ndcg10_list )
print("Average in NDCG@10 is %s"%ndcg10_avg)
print("Standard Deviation is %s"%std10)
