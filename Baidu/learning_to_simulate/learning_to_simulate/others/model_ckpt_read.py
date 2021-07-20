#import tensorflow as tf

# import tensorflow.compat.v1 as tf
# #通过NewCheckpointReader读取checkpoint中所有保存的变量
# reader = tf.train.NewCheckpointReader('/tmp/models/WaterRamps_10/model.ckpt')
# #获取一个从变量名到变量维度的字典类型的所有变量列表
# golbal_varibles = reader.get_variable_to_shape_map()

# for variable_name in golbal_varibles :
#     print(variable_name , golbal_varibles[variable_name])

#获取变量空间下名称为v1的变量的取值
#print('value for v1 is',reader.get_tensor('test/v1'))
import tensorflow as tf
import pprint # 使用pprint 提高打印的可读性
# NewCheck =tf.train.NewCheckpointReader('D:\Pastgraduate\Code\Others\deepmind\\tmp\models\WaterRamps_10')
NewCheck =tf.train.NewCheckpointReader('./WaterRamps_10/model.ckpt-10')   # ./model/model.ckpt
# print("debug_string:\n")
# pprint.pprint(NewCheck.debug_string().decode("utf-8"))

print(NewCheck.debug_string().decode("utf-8"))