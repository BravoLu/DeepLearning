# Tensorflow Learning

* 这里记录了Tensorflow学习中遇到的问题以及Tensorflow相关练习的源码

## saver&checkpoint
```python 
# revised in 2019-1-16
# checkpoint file
# model.ckpt-200.data-00000-of-00001
# model.ckpt-200.index
# model.ckpt-200.meta

# model.ckpt-200.meta ---- save the architecture of graph
# save
saver.save(sess,'my-model',global_step=step, write_meta_graph=False)
# load
tf.train.import_meta_graph('model.ckpt-200.meta')

# model.ckpt-200.data-00000-of-00001 ---- save weights,biases,operators

# model.ckpt-200.index  --- a nonchangable string list

# e.g about save model
# create saver
saver = tf.train.Saver(tf.global_variables(),max_to_keep=1) # max_to_keep gurantee only save the last data

checkpoint_path = os.path.join(Path,'model.ckpt')
saver.save(session,checkpoint_path,global_step=step) # step is the times of iteration

# e.g about load model
saver = tf.train.Saver(tf.global_variables())
model_file = tf.train.latest_checkpoint('PATH')
saver.restore(sess, model_file) 

```

## API
```python
tf.image.sample_distorted_bounding_box(
	image_size,
	bounding_boxes,
	seed=None,
	seed2=None,
	min_object_covered=0.1,
	aspect_ratio_range=None,
	area_range=None,
	max_attempts=None,
	use_image_if_no_bounding_boxes=None,
	name=None
	)

Generate a single randomly distorted bounding box for an image

return begin,size and bbox, the first two tensor can be fed directly into tf.slice to crop the image, the latter may be suppplied to tf.image.draw_bounding_boxes to visualize what the bounding box looks like.

arg:
bounding_boxes 
bounding_boxes = tf.constant([[[0. 0. 1. 1.]]]) -> [1,1,4]
```
https://www.jianshu.com/p/05c4f162c7e
```python

"""
input matrix: WxW
filter      : FxF
stride      : S
output matrix: new_height, new_width
"""

padding = 'VALID'

new_height = new_width = ceil((W - F + 1)/S)

padding = 'SAME'

new_height = new_width = ceil((W/S))

```
## 2019-1-17
```python
tf.add_to_collection
tf.get_collection
tf.add_n
tf.squeeze(input,axis=None,name=None,squeeze_dims=None)
tf.slice(input_,begin,size,name=None)
tf.control_dependencies() #use to control the sequence of operator
```