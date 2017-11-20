#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl = ros_to_pcl(pcl_msg)

    # Outlier filter
    out_filter = pcl_data.make_statistical_outlier_filter()
    out_filter.set_mean_k(10)
    out_filter.set_std_dev_mul_thresh(0.3)
    # Implement outlier filter
    sof = out_filter.filter()

    # TODO: Voxel Grid Downsampling   
    vox_filter = sof.make_voxel_grid_filter()

    LEAF = 0.005

    vox_filter.set_leaf_size(LEAF, LEAF, LEAF)

    vox = vox_filter.filter()

    # TODO: PassThrough Filter
    passthrough = vox.make_passthrough_filter()
    passthrough.set_filter_field_name('z')
    passthrough.set_filter_limits(0.6, 1.1) #Lesson 3: Part 11
    pt = passthrough.filter() 
    # May want second Pass through

    # TODO: RANSAC Plane Segmentation
    ransac = pt.make_segmenter()
    ransac.set_model_type(pc1.SACMODEL_PLANE)
    ransac.set_method_type(pc1.SAC_RANSAC)

    max_distance = 0.01 #from Lesson 3: Part 15
    ransac.set_distance_threshold(max_distance)

    # TODO: Extract inliers and outliers
    inliers, coefficients = ransac.segment()
    pcl_table = ransac.extract(inliers, negative=False)
    pcl_targets = ransac.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    gray_cloud = XYZRGB_to_XYZ(pcl_targets)
    tree = gray_cloud.make_kdtree()

    euc_c = gray_cloud.make_EuclideanClusterExtraction()

    euc_c.set_ClusterTolerance(0.02) #0.01
    euc_c.set_MinClusterSize(40) #25
    euc_c.set_MaxClusterSize(4000) #10000

    euc_c.set_SearchMethod(tree)

    clusters = euc_c.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_colors = get_color_list(len(clusters))
    cluster_colors_list = []

    for i, index in enumerate(clusters):
        for j, idx in enumerate(index):
            x = gray_cloud[idx][0]
            y = gray_cloud[idx][1]
            z = gray_cloud[idx][2]
            c = rgb_to_float(cluster_color[i])
            cluster_colors_list.append([x, y, z, c])

    cluster_pcl = pcl.PointCloud_PointXYZRGB()
    cluster_pcl.from_list(cluster_colors_list)


    # TODO: Convert PCL data to ROS messages
    ros_targets = pcl_to_ros(pcl_targets)
    ros_table = pcl_to_ros(pcl_table)
    ros_cluster = pcl_to_ros(cluster_pcl)


    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_targets)
    pcl_table_pub.publish(ros_table)
    pcl_cluster_pub.publish(ros_cluster)

# Exercise-3 TODOs: 
    print("here")
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects = []
    detected_objects_labels = []

    for index, pts_list in enumerate(clusters):

        # Grab the points for the cluster
        pcl_cluster = pcl_targets.extract(pts_list)

        # Convert Cluster to ROS from PCL
        ros_pcl_array = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_pcl_array, using_hsv=True)
        normals = get_normals(ros_pcl_array)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(gray_cloud[pts_list[0]])
        label_pos[2] += 0.4
        object_markers_pub.publish(make_label(label, label_pos, index)) 
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    print(detected_objects)
    
    if detected_objects:
        # Publish the list of detected objects
        detected_objects_pub.publish(detected_objects)

        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    else:
        ros.loginfo("No objects detected")

if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous = True)


    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size = 1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
