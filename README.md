# Face Recognition
A face recognition model using the facebook embedder model with dlib. 
An lasagne MLP is used for distinguishing between faces in the database.

[![Watch the video](https://github.com/JamesUnicomb/face_recognition/blob/master/screenshot.png)](https://www.youtube.com/watch?v=aa7Rk-uxxjI)
Video Link: https://www.youtube.com/watch?v=aa7Rk-uxxjI

## Face messages
For each processed frame a message will be published with a ML predicted face and probability accociated with that matching.
```
Header header
string[] names
float64[] probability
```
