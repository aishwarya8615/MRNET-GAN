from load_data_tf import MRNetDataset
obj = MRNetDataset("/home/vignesh/PycharmProjects/AI_Course/Project/mrnet_gan_project/MRNet-v1.0/", "acl", "axial")
ar1, l, w = obj.__getitem__(200)
print "array 1 is ", ar1.shape
print "label is ", l
print "weight is ", w