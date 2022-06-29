def findThreshold(model_name, distance_metric):

	base_threshold = {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75}

	thresholds = {
		'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86},
        'Facenet':  {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
        'Facenet512':  {'cosine': 0.30, 'euclidean': 23.56, 'euclidean_l2': 1.04},
        'ArcFace':  {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
        'Dlib': 	{'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.4},

        #TODO: find the best threshold values
        'SFace': 	{'cosine': 0.5932763306134152, 'euclidean': 10.734038121282206, 'euclidean_l2': 1.055836701022614},

		'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
		'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
		'DeepID': 	{'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17}

		}

	threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

	return threshold