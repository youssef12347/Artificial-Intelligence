import tensorflow as tf
import time
import numpy as np


if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('GPUs found, this will be fun!')
    
size = 10000
a = tf.random.uniform(shape=[size,size])
b = tf.random.uniform(shape=[size,size])

start = time.time()
c = tf.matmul(a,b)
print(time.time()-start)

execution_times_cpu = {
    'cpu_1x' : 35.3118622303009,
    'cpu_2x' : 15.228885173797607,
    'cpu_4x' : 10.392901182174683,
    'cpu_8x' : 6.591029644012451,
    'cpu_16x' : 2.9781200885772705  
}
execution_times_cpu

execution_times_gpu = {
    'gpu_1x' : 0.0012249946594238281,
    'gpu_2x' : 0.0007076263427734375,
    'gpu_4x' : 0.0006804466247558594   
}
execution_times_gpu

execution_times_cpu_gpu = {}
execution_times_cpu_gpu.update(execution_times_cpu)
execution_times_cpu_gpu.update(execution_times_gpu)
execution_times_cpu_gpu

max_time_cpu = np.max(list(execution_times_cpu.values()))
max_time_gpu = np.max(list(execution_times_gpu.values()))
max_time_cpu_gpu = np.max(list(execution_times_cpu_gpu.values()))

execution_times_norm_cpu = np.array(1)/(np.array(list(execution_times_cpu.values()))/max_time_cpu)
execution_times_norm_gpu = np.array(1)/(np.array(list(execution_times_gpu.values()))/max_time_gpu)
execution_times_norm_cpu_gpu = np.array(1)/(np.array(list(execution_times_cpu_gpu.values()))/max_time_cpu_gpu)

import seaborn as sns
sns.barplot(x=np.array(list(execution_times_cpu.keys())), y=execution_times_norm_cpu).set(ylabel='speedup')
sns.barplot(x=np.array(list(execution_times_gpu.keys())), y=execution_times_norm_gpu).set(ylabel='speedup')
sns.barplot(x=np.array(list(execution_times_cpu_gpu.keys())), y=execution_times_norm_cpu_gpu).set(ylabel='speedup')