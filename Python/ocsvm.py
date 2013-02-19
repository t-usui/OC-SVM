import sys
import numpy as np
import pylab as pl
import matplotlib.font_manager
from sklearn import svm

def parse_data(line):
	elements = None
	class_number = None
	dimensions = []
	values = []

	print line

	elements = line.split(" ")
	for element in elements:
		list = element.split(":")
		if len(list) == 1:
			class_number = int(list[0])
		elif len(list) == 2:
			dimensions.append(int(list[0]))
			values.append(float(list[1]))
	
	return class_number, dimensions, values


def construct_feature_vector(dimensions, values):
	feature_vector = []

	count = 0
	for i in range(0, 15):
		if i == dimensions[count]:
			feature_vector.append(values[count])
			if count < len(dimensions)-1:
				count += 1
		else:
			feature_vector.append(0)

	return feature_vector


def input_data(file_name):
	class_number_list = []
	dimensions_list = []
	values_list = []

	feature_vector_list = []

	file_read = open(file_name, "r")

	for line in file_read:
		class_number, dimensions, values = parse_data(line)
		class_number_list.append(class_number)
		dimensions_list.append(dimensions)
		values_list.append(values)

		feature_vector_list.append(construct_feature_vector(dimensions, values))

	return class_number_list, feature_vector_list

if __name__ == "__main__":
	training_class_number_list = None
	training_feature_vector_list = None
	test_class_number_list = None
	test_feature_vector_list = None

	argv = sys.argv
	argc = len(argv)

	if argc != 3:
		print "Error: argc"
		print "Usage: python ocsvm.py <training_file_name> <test_file_name>"
		sys.exit()

	training_file_name = argv[1]
	test_file_name = argv[2]

	training_class_number_list, training_feature_vector_list = input_data(training_file_name)
	test_class_number_list, test_feature_vector_list = input_data(test_file_name)

	for i in range(len(training_class_number_list)):
		print training_class_number_list[i],
		print training_feature_vector_list[i]

	for i in range(len(test_feature_vector_list)):
		print test_feature_vector_list[i]

	classifier = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.01)
	classifier.fit(training_feature_vector_list)
	result = classifier.predict(test_feature_vector_list)

	print result

	count = 0
	for i in result:
		if i == 1:
			count += 1
	
	print count

"""
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
"""
"""
pl.title("Novelty Detection")
pl.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=pl.cm.Blues_r)
a = pl.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
pl.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

b1 = pl.scatter(X_train[:, 0], X_train[:, 1], c='white')
b2 = pl.scatter(X_test[:, 0], X_test[:, 1], c='green')
c = pl.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
pl.axis('tight')
pl.xlim((-5, 5))
pl.ylim((-5, 5))
pl.legend([a.collections[0], b1, b2, c],
          ["learned frontier", "training observations",
		             "new regular observations", "new abnormal observations"],
					           loc="upper left",
							             prop=matplotlib.font_manager.FontProperties(size=11))
pl.xlabel(
    "error train: %d/200 ; errors novel regular: %d/20 ; "
	    "errors novel abnormal: %d/20"
		    % (n_error_train, n_error_test, n_error_outliers))
pl.show()
"""
