#!/usr/bin/env python

import numpy as np

from face_recognition import FaceRecognition

fr = FaceRecognition(load_model = False)

fr.train_network()
