This is just pet project.

Task:
Each fingerprint is written in individual file.
Part of file name is id of person.
Each fingerprint is 400 float vector.
Input is two fingerprints.
Output is 1 if fingerprints belong to one person, 0 otherwise.

Implementation:
Keras (tensorflow) network based on Dense layers is used.
Two stage learning:
1) First network provides classification of one fingerprint. Output is id of person.
2) Second network use part of first network. It takes 2 fingerprints, each is encoded by first network, then both representations are combined and binary classification is provided.
