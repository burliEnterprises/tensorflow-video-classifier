import tensorflow as tf
import sys
import os
import cv2
import math

# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

video_path = sys.argv[1]
# angabe in console als argument nach dem aufruf  


# holt labels aus file in array 
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!
				   
# graph einlesen, wurde in train.sh -> call retrain.py trainiert
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
    graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
	
	#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276
with tf.Session() as sess:

	video_capture = cv2.VideoCapture(video_path) 
	#frameRate = video_capture.get(5) #frame rate
	i = 0
	while True:  # fps._numFrames < 120
		frame = video_capture.read()[1]
		frameId = video_capture.get(1) #current frame number
		#if (frameId % math.floor(frameRate) == 0):
		if (0 == 0):
			i = i + 1
			cv2.imwrite(filename="screens/"+str(i)+"alpha.png", img=frame);
			image_data = tf.gfile.FastGFile("screens/"+str(i)+"alpha.png", 'rb').read()
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
			predictions = sess.run(softmax_tensor, \
					 {'DecodeJpeg/contents:0': image_data})		
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			for node_id in top_k:
				human_string = label_lines[node_id]
				score = predictions[0][node_id]
				print('%s (score = %.5f)' % (human_string, score))
			print ("\n\n")
			cv2.imshow("image", frame)
			cv2.waitKey(1)

	video_capture.release()
	cv2.destroyAllWindows()