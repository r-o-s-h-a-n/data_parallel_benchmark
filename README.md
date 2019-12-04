# data_parallel_benchmark

In recent years, deep learning models have become state-of-the-art models for learning from increasing volume and complexity of data. Deep learning models require large volumes of data to extract useful features. This process can require large amounts of time to train -- on the order of weeks. Because of the repetitiveness of some of the computations, distributed computing has presented value in this domain. 

There exist multiple open-source data parallel deep learning implementations. Popular/recent options:
* TensorFlow 2.0.0 includes tf.distribute.Strategy. MirroredStrategy
* PyTorch 1.2.0 includes torch.nn.DataParallel

In this project, we vary the dataset size, model size, batch size, and number of GPUs and train using two data parallel deep learning frameworks: PyTorch DataParallel and TensorFlow MirroredStrategy. We observe TensorFlow has higher processing rates and increased scaleup, but recognize a fairer comparison would be between TensorFlow MirroredStrategy and PyTorch DistributedDataParallel. We observe that GPUs improve performance when models have many parameters and batch size is high. 
