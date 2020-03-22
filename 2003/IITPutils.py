import tensorflow as tf , numpy as np
import re 
from tensorflow.contrib.layers import *
import sys
sys.path.append('/home/advice/Python/SR/Custom/')
from Activations import *

def generate_noise(samples , n_dim):
    '''
    Generate `samples` samples of uniform noise in 
    ([-1,1], [-1,1])
    '''
    return np.random.uniform(-1, 1, (samples, n_dim))

def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))

def OneHotIndex(in_var , num_var , one_hot_var) :
    start_idx = 0
    key_store = {}
    store = []
    for idx , col in enumerate(in_var) :
        if col in num_var :
            aa = [start_idx , start_idx +1]
            store.append(aa)
            start_idx += 1
        else :
            find =(
                [idx for idx , ck in enumerate(one_hot_var) 
                 if re.search("^{}_".format(col) , ck)]
                  )
            nn = len(find)
            aa = [start_idx , start_idx + nn]
            start_idx += nn
            store.append(aa)
        key_store[col] = aa
    return key_store , store


def get_weight_variable(shape, name=None,
                        type='xavier_uniform', regularize=True, **kwargs):
    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.get_variable(name, initializer=initial)
        else:
            weight = tf.get_variable(name, shape=shape, initializer=initial)
    if regularize:
        tf.add_to_collection('weight_variables', weight)
    return weight 


def UpConvOutputSize(input , output_channel_size , 
                     data_format , 
                     filter_size_h , filter_size_w , 
                     stride_h , stride_w , padding) :
    """calculation of the output_shape"""
    if data_format == "NHWC":
        input_channel_size = input.get_shape().as_list()[3]
        input_size_h = input.get_shape().as_list()[1]
        input_size_w = input.get_shape().as_list()[2]
        stride_shape = [1, stride_h, stride_w, 1]
        if padding == 'VALID':
            output_size_h = (input_size_h - 1)*stride_h + filter_size_h
            output_size_w = (input_size_w - 1)*stride_w + filter_size_w
        elif padding == 'SAME':
            output_size_h = (input_size_h - 1)*stride_h + 1
            output_size_w = (input_size_w - 1)*stride_w + 1
        else:
            raise ValueError("unknown padding")
        output_shape = tf.stack([tf.shape(input)[0], 
                            output_size_h, output_size_w, 
                            output_channel_size])
    elif data_format == "NCHW":
        input_channel_size = input.get_shape().as_list()[1]
        input_size_h = input.get_shape().as_list()[2]
        input_size_w = input.get_shape().as_list()[3]
        stride_shape = [1, 1, stride_h, stride_w]
        if padding == 'VALID':
            output_size_h = (input_size_h - 1)*stride_h + filter_size_h
            output_size_w = (input_size_w - 1)*stride_w + filter_size_w
        elif padding == 'SAME':
            output_size_h = (input_size_h - 1)*stride_h + 1
            output_size_w = (input_size_w - 1)*stride_w + 1
        else:
            raise ValueError("unknown padding")
        output_shape = tf.stack([tf.shape(input)[0], output_channel_size, 
                                 output_size_h, output_size_w])
    else:
        raise ValueError("unknown data_format")
    return [input_channel_size , output_shape , stride_shape]

def upconvolution(input, output_channel_size, 
                  filter_size_h, filter_size_w,
                  stride_h, stride_w, type , layer_name, 
                  bn = True , is_training = True , activation = tf.nn.relu, 
                  data_format="NHWC", padding='VALID' , 
                  init_b = tf.random_normal_initializer(stddev=0.01)):
    """
    Type : 
    xavier_uniform / xavier_normal / he_normal
    he_uniform     / he_uniform    / caffe_uniform
    """
    with tf.variable_scope(layer_name):
      #creating weights:
        shapelists = UpConvOutputSize(input , output_channel_size , 
                         data_format , filter_size_h , filter_size_w , 
                         stride_h , stride_w , padding)
        input_channel_size , output_shape , stride_shape = shapelists
        shape = [filter_size_h, filter_size_w, 
               output_channel_size, input_channel_size]
        W_upconv = get_weight_variable(shape, name="w",
                                       type=type, regularize=True)
        shape=[output_channel_size]
        b_upconv = tf.get_variable("b", shape=shape, dtype=tf.float32, 
                                 initializer=init_b)
#         tf.summary.histogram(f"Weight{layer_name}", W_upconv)
        tf.add_to_collection('weight_variables', W_upconv)
        upconv = tf.nn.conv2d_transpose(input, W_upconv, output_shape, stride_shape,
                                      padding=padding,
                                      data_format=data_format)
        output = tf.nn.bias_add(upconv, b_upconv, data_format=data_format)
        #Now output.get_shape() is equal (?,?,?,?) which can become a problem in the 
        #next layers. This can be repaired by reshaping the tensor to its shape:
        output = tf.reshape(output, output_shape)
        print(f"{layer_name:10} output : {output.get_shape().as_list()}")
        if bn == True : 
            output = tf.layers.batch_normalization(output , training = is_training)
        #now the shape is back to (?, H, W, C) or (?, C, H, W)
        return activation(output)
    

def spectral_norm(w, iteration=1 , name = None):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable(name , [1, w_shape[-1]], 
                        initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm 

def select_init(activation) :
    select_w_init = np.random.randint(0, 2, size=1 )[0]
    seed_n = np.random.randint(1, 2000, size=1 )[0]
    relu_w_init = [tf.keras.initializers.he_uniform(seed = seed_n) ,
                   tf.keras.initializers.he_normal(seed = seed_n)][select_w_init]
    tanh_w_init = [tf.keras.initializers.glorot_normal(seed = seed_n) ,
                   tf.keras.initializers.glorot_uniform(seed = seed_n)][select_w_init]
    s_elu_w_init = [tf.keras.initializers.lecun_normal(seed = seed_n) ,
                   tf.keras.initializers.lecun_uniform(seed = seed_n)][select_w_init]
    nomal_w_init = tf.keras.initializers.truncated_normal(seed = seed_n)
    if activation in [tf.nn.leaky_relu, tf.nn.relu] :  
        init = relu_w_init
    elif activation in [tf.nn.tanh , tf.nn.softmax] :  
        init = tanh_w_init
    elif activation in [tf.nn.selu , tf.nn.elu , mish] :      
        init = s_elu_w_init
    else : 
        init = nomal_w_init
    return init 

def LayerByType(layer, Type , activation) :
    if Type == "SWA" :
        layer = moving_free_batch_norm(layer, axis=-1, 
                                       training=is_training_bn,
                                       use_moving_statistics=use_moving_statistics, 
                                       momentum=0.99)
    elif Type == "Self_Normal" :
        if activation == nalu :
            layer = activation(layer ,2 , name = "NALU_" + name )
        else :
            layer = activation(layer)
        layer = tf.contrib.nn.alpha_dropout(layer , 0.8)
    elif Type == "Batch_Normalization" :
        layer = tf.contrib.layers.batch_norm(layer, 
                                             center=True, scale=True, 
                                             is_training=True, # phase
                                             scope='bn')
    elif Type == "Instance_Normalization" :
        layer = tf.contrib.layers.instance_norm(layer)
    else : pass
    if Type == "Self_Normal" : 
        pass
    else : 
        layer = activation(layer)
    return layer

def UseBias(input , W , usebias , shape , init = tf.constant_initializer(0.0)) :
    layer = tf.matmul( input , W)
    if usebias :
        bias = tf.get_variable("Bias", 
                               shape = [shape[1]] , dtype = tf.float32 , 
                             initializer = init )
        layer += bias
    else :
        pass
    return layer
    
def fully_connected_layer(input , shape = None, 
                          name = None , activation = tf.nn.leaky_relu ,
                          usebias = True , final = False , 
                          SN = True , Type = None ) :
    with tf.variable_scope(name):
        input_size = input.get_shape().as_list()[1]
        shape = [input_size , shape]
        init = select_init(activation)
        W1 = tf.get_variable(f"Weight", dtype = tf.float32 , 
                             shape = shape , initializer = init)
        tf.add_to_collection('weight_variables', W1)
        if SN :
            W1 = spectral_norm(W1 , name =f"SN_Weight")
        if final :
            layer = UseBias(input=input , W=W1 , usebias=usebias , shape =shape)
        else :
            layer = UseBias(input=input , W=W1 , usebias=usebias , shape =shape)
            layer = LayerByType(layer, Type , activation)
        print(f"{name:10} output : {layer.get_shape().as_list()}")
        return layer
    
def VariableHandling(input , key_store) :
    x_input = []
    r_input = []
    arg_input = []
    for a, which in key_store.items() :
        start , terminal = which[0] , which[1]
        diff = terminal - start 
        x_split = tf.slice(input, [0, start] , [-1 , diff])
        if diff == 1 : 
            xx = tf.nn.tanh(x_split)
            real_x = xx
            arg_input.append(xx)
        else : 
            print(f"{a} / diff : {diff}")
            xx = tf.nn.softmax(x_split)
            arg_max = tf.argmax(xx , axis = 1 )
            arg_max2 = tf.reshape(tf.cast(arg_max,tf.float32),
                                  (-1,1))
            #arg_max = tf.cast(arg_max , tf.int32)
            real_x = tf.one_hot(arg_max , depth=diff)
            arg_input.append(arg_max2)
        x_input.append(xx)
        r_input.append(real_x)
    return [x_input , r_input , arg_input]

def CreateMissIndicator(data , store ,  num_var) :
    cat_indicater = []
#     data = pd.DataFrame(data, columns = x_var)
    indi_finder = data
    for c in store :
        diff = c[1] - c[0]
        if diff ==1 : 
            pass
        else :
            ck = indi_finder.iloc[:,c[0]:c[1]].sum(axis = 1)
            list_ = ck[ck==0].index.tolist()
            a = np.ones((len(ck), diff))
            a[list_,:] = 1
            cat_indicater.append(a)
    del indi_finder
    num_shape = data[num_var].shape
    num_miss_indicator = np.ones(shape=(num_shape[0], num_shape[1]))
    num_miss_indicator[np.isnan(data[num_var])] = 1.0
    if cat_indicater == [] :
        total_indicator = num_miss_indicator
    else :
        cat_miss_indicater = np.concatenate(cat_indicater , axis = 1)
        total_indicator = np.concatenate([num_miss_indicator ,
                                          cat_miss_indicater] , axis = 1)
    return total_indicator

def CatNumEmb( out  , store ) :
        condition = store
        inputs = []
        args   = []
        for key , cond in condition.items() :
            first , to = cond[0] , cond[1]  
            diff = to - first
            split = tf.slice(out , [0 , first ],
                             [ -1 , diff ] ) # 
            if diff == 1 :
                __split__ = tf.nn.tanh(split)
                arg   = tf.nn.tanh(split)
                inputs.append(__split__)
                args.append(arg)
            else :
                __split__ = tf.nn.softmax(split)
                arg   = tf.expand_dims(tf.argmax(first , axis = 1 ) ,axis = 1 )
                arg   = tf.cast(arg , dtype = tf.float32)
                inputs.append(__split__)
                args.append(arg)
        Input = tf.concat(inputs, axis=1, name='Inputs')
        Arg = tf.concat(args, axis=1, name='Args')
        return Input , Arg

def MissGeneratorByVar( input  , overall_where) :
        inputs = []
        numinputs = []
        for idx in range(len(overall_where)) :
            cond = overall_where[idx]
            first , to = cond[0] , cond[1]  
            diff  = to - first
            if idx == 0 :            
                ## nuermic 모으기
                split = tf.slice(input , [0 , first ] ,[ -1 , diff] ) # 
                inputs.append(split)
                numinputs.append(split)
                numfinal = diff
            else :
                split = tf.slice(input , [0 , numfinal ] ,[ -1 , 1 ] ) # 
                split = tf.tile(split , [1,diff])
                split2 = tf.zeros_like(split)
                numfinal +=1
                inputs.append(split)
                numinputs.append(split2)
        MissGenerator = tf.concat(inputs, axis=1, name='Inputs')
        NumMissGenerator = tf.concat(numinputs, axis=1, name='Inputs')
        return MissGenerator , NumMissGenerator
    
def missing_handling(result , overall_where) :
    totalresult = []
    for idx , cond in enumerate(overall_where) :
        start , end = cond[0] , cond[1]
        if idx == 0 :
            result[:,start:end] = np.where(result[:,start:end]==1.5, 
                                           np.nan, 
                                           result[:,start:end]) 
            totalresult.append(result[:,start:end])    
        else :
            idx = np.sum(result[:,start:end] , axis = 1) == 0
            result2 = np.argmax(result[:,start:end] , axis = 1).astype(np.float32)
            result2[idx] = np.nan
            result2 = np.reshape(result2,(-1,1))
            totalresult.append(result2)  
    return np.concatenate(totalresult,axis=1)


def handling_column_index(onehot_key_store:dict) -> list :
    """
    onehot_key 연속형 변수 하나의 인덱스로 모으기
    """
    overall_where = []
    for key , values in onehot_key_store.items() :
        first , to = values
        diff = to-first
        if diff == 1 :
            final = to
            continue
        else :
            if len(overall_where) == 0 :
                overall_where.append([0,final])
                overall_where.append(values)
            else :
                overall_where.append(values)
    return overall_where