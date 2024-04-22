import numpy
import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import optimizers
from keras.models import  Sequential,Model
from keras.layers import Dropout,Flatten,Dense,GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard,EarlyStopping
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense


files_train = 0 # 训练样本数
files_validation = 0 # 测试样本数

cwd = os.getcwd() # 获取当前工作目录的路径
folder = 'train_data/train'
for sub_folder in os.listdir(folder): # 遍历目标文件夹 folder 中的每个子文件夹
    path,dirs,files = next(os.walk(os.path.join(folder,sub_folder))) # 当前子文件夹的路径、子文件夹列表和文件列表
    files_train += len(files)

folder = 'train_data/test'
for sub_folder in os.listdir(folder):
    path,dirs,files = next(os.walk(os.path.join(folder,sub_folder)))
    files_validation += len(files)

print(files_train,files_validation)

img_width, img_height = 48,48
train_data_dir = "train_data/train"
validation_data_dir = "train_data/test"
nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2

# 加载预训练模型
# applications.VGG16：这是 Keras 库中的一个函数，用于加载预训练的 VGG16 模型。VGG16 是一个经典的卷积神经网络模型，适用于图像分类任务
# weights='imagenet'：这个参数指定了模型加载的权重来源，'imagenet' 表示使用在 ImageNet 数据集上预训练好的权重
# include_top=False：这个参数指定是否包含模型的顶层（全连接层）
model = applications.VGG16(weights='imagente',include_top=False, input_shape=(img_width,img_height,3))

# 冻结前10层，保持它们在训练过程中的参数不变，
# 使得模型在新任务上更快地收敛，并且能够更好地利用已经学到的特征
for layer in model.layers[:10]:
    layer.trainable = False

x = model.output # 输出通常是一个张量
x = Flatten()(x) # 将模型的输出展平为一维向量，以便连接到全连接层等层中

# 创建了一个全连接层，将展平后的向量 x 通过全连接层，得到每个类别的概率分布
# num_classes 是输出的类别数量
# activation="softmax" 指定了该层的激活函数为 softmax 函数
# softmax 激活函数确保了输出的每个元素都是在 [0, 1] 范围内，并且所有元素的总和为 1，表示各个类别的预测概率
predictions = Dense(num_classes, activation="softmax")(x)

# 创建了一个新的神经网络模型
# 指定了新模型的输出层为predictions，为了将之前定义的模型输出与新模型相连接，使新模型的输出是通过原模型的输出层后的结果
model_final = Model(input = model.input, output = predictions)

# 编译了最终的神经网络模型 model_final，指定了模型的损失函数、优化器和评估指标，为模型的训练做准备
# categorical_crossentropy: 损失函数是交叉熵损失函数，用于衡量模型预测的概率分布与真实标签之间的差异
# SGD: 随机梯度下降优化器，并设置了学习率为 0.0001 和动量参数为 0.9。
# 优化器的作用是根据损失函数的梯度来更新模型的参数，使得损失函数的值逐渐减小，从而提高模型的性能
# accuracy: 指定了准确率为模型在训练和测试过程中需要监测的指标，用于衡量模型在预测时的准确性
model_final.compile(loss = "categorical_crossentropy",
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])

# 定义了一个图像数据生成器 train_datagen，用于对训练数据进行数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,             # 将图像的像素值缩放到 [0,1] 范围内，以便更好地进行训练
    horizontal_flip=True,      # 随机水平翻转图像，增加训练样本的多样性
    fill_mode="nearest",       # 在进行图像变换时，使用最近邻插值填充图像空白区域
    zoom_range=0.1,            # 随机缩放图像的范围，可以使图像在训练过程中具有不同的缩放程度
    width_shift_range=0.1,     # 随机水平平移图像的范围，可以使图像在训练过程中具有不同的平移程度
    height_shift_range=0.1,    # 随机垂直平移图像的范围，可以使图像在训练过程中具有不同的平移程度
    rotation_range=5           # 随机旋转图像的角度范围，可以使图像在训练过程中具有不同的旋转角度
)

test_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip=True,
    fill_mode = "nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5
)

# 从指定目录中读取图像数据，并进行数据增强操作，
# 然后生成用于训练的批量图像数据和对应的标签
train_generatior = train_datagen.flow_from_directory(
    validation_data_dir, # 存储验证数据的目录路径
    target_size=(img_height,img_width), # 指定了生成的图像数据的尺寸
    class_mode="categorical" # 指定了分类类型
)

validation_generator = test_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# 定义了一个模型检查点 checkpoint，用于在训练过程中保存模型的最佳权重
# 模型在训练过程中会监视验证集上的准确率指标，只有在验证集上的准确率提高时才会保存模型的权重，并且每个训练周期都会保存一次检查点
# "car1.h5": 这是指定保存模型权重的文件名
# 'val_acc': 使用验证集的准确率（validation accuracy）作为监视指标
# verbose=1: 控制着日志输出的详细程度。设置为 1 表示输出保存模型的日志信息
# save_best_only=True: 只有当监视指标（在这里是验证集准确率）有所提高时才保存模型
# save_weights_only=False: 设置为 False，保存整个模型，包括结构、权重和优化器状态
# mode='auto': 指定了监视指标的模式为 'auto' 表示根据监视指标的类型自动选择监视模式
# period=1: 设置为 1 表示每个训练周期都保存一次检查点
checkpoint = ModelCheckpoint("car1.h5",monitor='val_acc',verbose=1,save_best_only=True,
                             save_weights_only=False,mode='auto',period=1)

# 定义了一个早期停止（Early Stopping）的回调函数 early，用于在验证集上的性能停止提升时提前停止训练
# 有助于防止模型过拟合，并在训练过程中节省时间和计算资源
# 'val_acc': 使用验证集的准确率（validation accuracy）作为监视指标
# min_delta=0: 如果监视指标的变化量小于或等于 min_delta，则被认为没有提升
# patience=10: 设置为 10 表示如果在连续 10 个训练周期内监视指标没有提升，则停止训练
# verbose=1: 设置为 1 表示输出早期停止的日志信息
# mode='auto': 表示根据监视指标的类型自动选择监视模式
early = EarlyStopping(monitor='val_acc',min_delta=0,patience=10,verbose=1,mode='auto')

# 使用 fit_generator 方法来训练模型 model_final，
# 并指定了训练过程中要使用的数据生成器、训练样本数量、训练周期数、验证数据生成器以及验证样本数量。
# 同时，还传递了两个回调函数 checkpoint 和 early 作为训练过程中的回调函数
history_object = model_final.fit_generator(
    train_generatior, # 生成器对象，用于在训练过程中生成训练数据的批次
    samples_per_epoch = nb_train_samples, # 每个训练周期中使用的训练样本总数
    epochs = epochs, # 指定了训练的总周期数，即整个训练过程会重复多少次
    validation_data=validation_generator, # 用于验证模型性能的数据
    nb_val_samples = nb_validation_samples, # 用于验证的样本总数
    callbacks=[checkpoint,early]

)














