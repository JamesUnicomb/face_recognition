# Face Recognition
A face recognition model using the facebook embedder model with dlib. 
A lasagne MLP is used for distinguishing between faces in the database.

[![Play the gif](https://github.com/JamesUnicomb/face_recognition/blob/master/faces_encoded.gif)](https://www.youtube.com/watch?v=aa7Rk-uxxjI)
Video Link: https://www.youtube.com/watch?v=aa7Rk-uxxjI

## Face Messages
For each processed frame a message will be published with a ML predicted face and probability accociated with that matching.
```
Header header
string[] names
float64[] probability
float64[] distances
```
Probabilities and distances for each prediction are given for each face measurement for rejection of unknowns.

## Adapting to your Database
In the face_recognition/training_data folder, make a list of folders of people you want your algorithm to detect.

For the current code, there are five people: Mark, Jez, Johnson, Sophie and Dobby. Put photos of each person in each of these folders (for your people).

Note: each image can only contain one person.

Once your database is complete, run the script:
```
python path_to_face_recognition/scripts/train_face_recognition.py
```

This will save a model in the face_recognition/models folder.

## Future Work
Work on a way of recognising unknown faces and rejecting faces not in the database.
