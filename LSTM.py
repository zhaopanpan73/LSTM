# wget http://www.fit.vvutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# tar xvf simple-examples.tgz
# git clone https://github.com/tensorflow/models.git
# cd models/tutorials/rnn/ptb
# import time

# To run:
# $ python ptb_word_lm.py --data_path=simple-examples/data/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


# 定义语言模型处理输入数据的Class  PTBInput
class PTBInput(object):
    def __init__(self,config,data,name=None):
        self.batch_size=batch_size=config.batch_size
        self.num_steps=num_steps=config.num_steps
        self.epoch_size=((len(data)//batch_size)-1)//num_steps   # 这里为什么要减一，不是很理解

        self.input_data.self.targets=reader.ptb_producer(data,batch_size,num_steps,name=name)

# 定义语言模型的class，PTBModel
class PTBModel(object):
    def __init__(self,is_training,config,input_):
        self._input=input_

        batch_size=input_.batch_size
        num_steps=input_.num_steps
        size=config.hidden_size
        vocab_size=config.vocab_size
    # 用tf.contrib.rnn.BasicLSTMCell设置我们默认的LSTM单元
    # 其中隐含节点数为前面提取的hidden_size，forget_bias（foret gate的bias）为0,state_is_tuple也为True,这代表接收和返回的state将是2-tuple的形式
    # 同时，如果在训练状态，而且Dropout的keep_prob小于1，则在前面的lstm_cell之后接一个Dropout层，这里的做法是调用tf.contri       b.rnn.DropoutWrapper函数。
    # 最后使用RNN堆叠函数tf.contrib.rnn.MultiRNNCell将前面构造的lstm_cell多层堆叠到cell。堆叠次数为config中的num_layers,这里同样将state_is_tuple设置为True
    # 并用cell.zero_state设置为LSTM单元的初始化状态设置为0,。
    # 这里需要注意，LSTM单元可以读入一个单词并结合之前的状态state计算下一个单词出现的概率分布（这个很重要），并且每次读取一个（这里是一个，看清楚）单词后它的状态state将会被更新。
        def lstm_cell(self):
            return tf.contrib.rnn.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)
        attn_cell=lstm_cell
        # 当lstm后面需要dropout操作的时候
        if is_training and config.keep_prob<1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(),output_keep_prob=config.keep_prob)
            cell=tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)],state_is_tuple=True)

            self._initial_state=cell.zero_state(batch_size,tf.float32)  # ---------------*************----------------------

        # 我们创建网络的次嵌入部分，embedding即将为one-hot的编码格式的单词转化为向量的表达形式，我们用with tf.device("/cpu:0")
        # 然后初始化embedding矩阵，其行数设置为词汇表的vocab_size，列数（即每个单词的向量表达维数）设为hidden_size,和LSTM单元中的隐含节点一致。在训练过程中，embedding的参数可以内优化和更新。接下来用tf.nn.embedding_lookup查询单词对应的向量表达获得inputs)
        # 同时，如果为训练状态则再添加一层Dropout
        with tf.device("/cpu:0"):
            embedding=tf.get_variable("embedding",[vocab_size,size],dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(embedding,input_.input_data)
        if is_training and config.keep_prob<1:
            inputs=tf.nn.dropout(inputs,config.keep_prob)    # 这个inputs是已经训练得到的输出吗，所以这里可以直接应用dropout里面

        # 接下里定义输出的outputs,我们先使用tf.variable_scope将接下来的操作的名称设为RNN，一般为了控制训练的过程，我们回限制梯度在反向传播的时可以展开的步数为一个固定值，而这个步数就是num_steps。在这里我们设置一个循环，循环的长度为num_steps，来控制梯度的出传播。
        # 并且从第二次循环开始，我们使用tf.get_variable_scope.reuse_variables设置复用变量。（这才是步数展开的具体实现的具体技术点）
        # 每次在询函内，我们传入inputs和state到堆叠的LSTM单元（即cell）当中。
        # 这里需要注意inputs有三个维度，第一个维度代表的是batch中的第几个样本，第二个维度代表的是样本中的第几个词，第三个维度是单词的向量表达的维度，而inputs[:,time_step，:]代表所有样本的第time_step个单词。这里我们得到输出cell_output和更新后的time_step。(不应该是更新后的权重吗，为什么是time_step呢)，最后我们将结果cell_output添加到输出列表outputs
        outputs=[]
        state=self._initial_state   # ------------------***************-----------------
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):# 对于每一个cell读取一个单词来进行训练权重
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                    (cell_output,state)=cell(inputs[:,time_step,:],state)
                    outputs.append(cell_output)  # 将每一个cell的结果异常存入到outputs列表中

        # 接下来将output的内容concat串联到一起，并且使用tf.reshape将其转换为一个很长的一维向量。接下来是softmax层，先定义权重和softmax_w和偏置softmax_b，然后使用tf.matmul将输出output乘上权重并加上偏置得到logits,即网络的最后的输出。
        # 然后定义损失的loss，这里直接使用tf.contrib.legacy_seq2seq.sequence_loss_by_example计算出logits和targets的偏差。
        # 这里的sequence_loss即target_words的average negative log probability，然后使用tf.reduce_mean汇总batch的误差，在计算平均到每个样本的误差cost
        # 并且我们保留最终状态为final_state。此时如果不是训练状态。则直接返回
        output=tf.reshape(tf.concat(outputs,1),[-1,size])  # szie 隐藏层的个数
        softmax_w=tf.get_variable("softmax_w",[size,vocab_size],dtype=tf.float32)
        softmax_b=tf.get_variable("softmax_b",[vocab_size],dtype=tf.float32)
        logits=tf.matmul(output,softmax_w)+softmax_b
        loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[tf.reshape(input_.targets,[-1])],[tf.ones([batch_size*num_steps],dtype=tf.float32)])
        self._cost=cost=tf.reduce_sum(loss)/batch_size
        self._final_state=state

        if not is_training:
            return

        # 下面定义学习速率的变量_lr，并将其设置为不可训练。再使用tf.trainable_variable获取全部可以训练的参数tvars。
        # 这里针对前面得到的cost，计算tvars的梯度，并用tf.clip_by_global_norm设置梯度的最大范数max_grad_norm。
        # 这既是Gradient Clipping的方法，控制梯度的最大范数，某种程度上起到了正则化的效果。防止梯度爆炸的问题。
        # 定义优化器为：GradientDescent
        # 然后定义优化器_trian_op，用optimizer.apply_gradients将前面clip过的梯度应用到所有可以训练的参数tvars上，
        # 然后使用tf.contrib.framework.get_or_create_global_step生成全局统一的训练步数。
        self._lr=tf.Variable(0.0,trainable=False)   # 原来对变量可以设置训不训练
        tvars=tf.trianable_variables()  # 这样就全部获得了可训练的参数。 # -------------************************----------------------
        # tf.trainable_variables返回的是需要训练的变量列表（只要变量在定义的时候没有trainable=False就算），tf.all_variables返回的是所有变量的列表
        grads,_=tf.clip_by_average_norm(tf.gradients(cost,tvars),config.max_grad_norm)  # 在这里设置的梯度裁剪
        optimizer=tf.trian.GradientDescentOptimizer(self._lr)
        self._trian_op=optimizer.apply_gradients(zip(grads,tvars),global_step=tf.contrib.framework.get_or_create_global_step()) # 将裁剪过的梯度用到所有可训练的参数tvars上

        # 这里设置一个名为_new_lr(new learning rate)的placeholder用以控制学习速率，同时，定义操作_lr_update，它使用tf.assign将_new_lr的值赋给当前的学习速率_lr，
        # 再定义一个assign_lr的函数，用来在外部控制模型的学习速率，方式是将学习速率值传入_new_lr这个placeholder，并执行_update_lr操作完成对学习率的修改
        # 上面说self._lr不可训练，但是可以通过session对其进行修改，人为控制_lr的变化
        self._new_lr=tf.placeholder(tf.float32,shape=[],name="new_learning_rate")  # 这个place_holder,可以通过session传入值
        self._lr_update=tf.assign(self._lr,self._new_lr)  # tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number
        # tf.assign(ref, value, validate_shape=None, use_locking=None,name=None)
        # ref: A mutable Tensor. Should be from a Variable node. May be uninitialized.
        # 这里有一个问题，这里没有对self._lr直接赋值，而是用了一个place_holder对传入传入新值，然后用用这个place_holder的值对原始的不可训练的self._lr进行赋值
        # 应该是为了通过session动态传值进行修改吧

        def assign_lr(self,session,lr_value):  # 函数的目的：通过在外部来控制模型的学习速率
            session.run(self._lr_update,feed_dict={self._new_lr:lr_value})

        # 至此，模型的定义部分就完成了。我们再定义一个PTBMdel class的一些property
        # 在Python中的@property装饰器可以将返回变量设置为只读，防止修改变量引发的问题。
        # 这里定义 input,initial_state,cost,final_state,lr,train_op为property，以便外部访问。

    @property
    def input(self):
        return  self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return  self._cost

    @property
    def final_state(self):
        return  self._final_state

    @property
    def lr(self):
        return  self._lr

    @property
    def train_op(self):
        return self._train_op




    # 接下来定义几种不同大小的模型参数。
    # 首先是小模型的参数，下面解释各个参数的含义
    # init_scale是昂罗中的权重值的初始scale.
    # learning_rate是学习速率的初始值
    # max_grad_norm即前面提到的梯度的最大范数
    # num_layers是LSTM堆叠的层数
    # num_steps 是LSTM反向传播的展开步数
    # hidden_size是LSTM内的隐含节点数
    # max_epoch是初始学习率可训练的epoch数，在此之后需要调整学习率
    # max_max_epoch是总共可训练的epoch
    # keep_prob是dropout层的保留节点的比例
    # lr_decay是学习速率的衰减速度
    # batch_size是每个batch中样本的数量。
    # 具体每个参数的值，在不配置中对比才有意义
    # 接下来我们会在几个配置中讨论具体数值
class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

# 下面定义训练一个epoch数据的函数run_epoch。我们记录当前的时间，初始化损失costs和迭代数iters,并执行model.initial_state来初始化状并取得初始状态
# 接着创建输出结果的字典表fetches，其中包括xost和final_state,如果有评测操作eval_op，也一并加入fetches。接着我们进入循环中，次数即为epoch_size。
# 在每次循环中，我们生成训练用的fect_dit，，将全部的LSTM单元的state加入feet_dict中，然后传入feet_dict并执行fetches对网络进行一次训练。并拿到cost和state.
# 这里我们每完成约10%的epoch，就进行一次结果的展示，一次展示当前的epoch的进度，perplexity（即平均cost的自然常数指数，是语言模型中用来比较模型性能的重要指标，越低代表模型输出的概率分布在预测样本上越好）和训练速度（单词数每秒）
# 最后返回perplexity作为函数的结果

def run_epoch(session,model,eval_op=None,verbose=False):
    start_time=time.time()
    costs=0.0
    iters=0
    # state=session.run(model.initial_state)  # 取得网络当前的状态，# 然后看这epoch_size
    # 从网络中要得到的数据
    fetches={"cost":model.cost,"final_state":model.final_state}
    if eval_op is not None:
        fetches["eval_op"]=eval_op
    for step in range(model.input.epoch_size):
        feed_dict={}  # 馈送进网络的数据
        for i ,(c,h) in enumerate(model.initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        # 我送入一次数据到一个cell,然后得到这个cell的代价和最终状态
        vals=session.run(fetches,feed_dict)
        cost=vals["cost"]
        state=vals["final_state"]
        cost+=cost
        iters+=model.input.num_steps  # LSTM向后展开的步数

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))

        return np.exp(costs / iters)

def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()















