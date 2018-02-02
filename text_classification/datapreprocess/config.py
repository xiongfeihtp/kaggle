import os
import tensorflow as tf
from prepro import prepro


flags = tf.flags
home = os.path.expanduser("~")
train_file = "./pre_train_none_clear.json"
test_file = "./pre_test_none_clear.json"
glove_word_file = "./glove.840B.300d.txt"

target_dir = "./data"

train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")

word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")

train_eval = os.path.join(target_dir, "train_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "train_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")




if not os.path.exists(target_dir):
    os.makedirs(target_dir)


flags.DEFINE_string("mode", "prepro", "Running mode train/debug/test")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")
flags.DEFINE_string("load_path",None,"retrain_path")
flags.DEFINE_integer("load_step",0,"retrain globel step")


flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")


flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 8, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")
flags.DEFINE_integer("label_dim", 6, "label_dim")
# Extensions (Uncomment corresponding code in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained character embedding")


fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")


def main(_):
    config = flags.FLAGS
    if config.mode == "prepro":
        prepro(config)
    else:
        print("Unknown mode")
        exit(0)
if __name__ == "__main__":
    tf.app.run()
