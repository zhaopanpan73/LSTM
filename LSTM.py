# wget http://www.fit.vvutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# tar xvf simple-examples.tgz
# git clone https://github.com/tensorflow/models.git
# cd models/tutorials/rnn/ptb
# import time
import numpy as np
import tensorflow as tf
import reader


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
        tvars=tf.trianable_variables()  # 这样就全部获得了可训练的参数。
        # tf.trainable_variables返回的是需要训练的变量列表（只要变量在定义的时候没有trainable=False就算），tf.all_variables返回的是所有变量的列表
        grads,_=tf.clip_by_average_norm(tf.gradients(cost,tvars),config.max_grad_norm)  # 在这里设置的梯度裁剪
        optimizer=tf.trian.GradientDescentOptimizer(self._lr)
        self._trian_op=optimizer.apply_gradients(zip(grads,tvars),global_step=tf.contrib.framework.get_or_create_global_step()) # 将裁剪过的梯度用到所有可训练的参数tvars上








