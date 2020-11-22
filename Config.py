#quant_method='tensorflow' 
quant_method='custom'

alpha=0.75
width_multiplier=0.75
depth_multiplier=0.75
data_dir="data/cifar-10-data"
job_dir="log"
train_batch_size=256
eval_batch_size=100
learning_rate=0.1
decay_rate=0.94
decay_per_epoch=4
num_epoch=5
weight_decay=1e-2
#1e-2, accuracy=0.84
#1e-3, accuracy=0.8291
#4e-5, accuracy=0.81xx
